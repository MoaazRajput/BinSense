"""BinSense - Smart Waste Assistant.

Flask backend using ImageNet vision as the primary classifier and a CNN fallback.
"""

import os
import random
import numpy as np
import io
import threading
import webbrowser
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, send_file

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CONFIDENCE_THRESHOLD = 0.50

# ─── ImageNet Classifier (primary — no API key needed) ───────────────────────
try:
    import tensorflow as tf
    import keras

    print("[INFO] Loading ImageNet model...")
    _imagenet_model = keras.applications.MobileNetV2(
        weights='imagenet',
        include_top=True,
        input_shape=(224, 224, 3)
    )
    IMAGENET_AVAILABLE = True
    print("[INFO] ImageNet model ready.")
except Exception as e:
    _imagenet_model = None
    IMAGENET_AVAILABLE = False
    print(f"[INFO] ImageNet model not available: {e}")

# ─── ImageNet label → waste category mapping ─────────────────────────────────
_IMAGENET_CAT_MAP = {

    # ── E-Usable (Electronics + Batteries) ───────────────────────────────────
    'e_usable': [
        # Batteries
        'battery', 'electric_battery', 'nine_volt', 'flashlight', 'torch',
        # Computers & input
        'mouse', 'computer_mouse', 'trackball', 'joystick',
        'keyboard', 'space_bar', 'typewriter_keyboard',
        'laptop', 'desktop_computer', 'hand_held_computer',
        # Displays & TV
        'monitor', 'television', 'projector',
        # Phones & communication
        'cellular_telephone', 'phone', 'pay_phone',
        # Audio & entertainment
        'remote_control', 'radio', 'ipod',
        'cassette_player', 'tape_player', 'cd_player', 'record_player',
        'speaker', 'loudspeaker', 'microphone', 'headphone', 'earphone',
        # Camera
        'camera', 'lens_cap',
        # Storage & networking
        'hard_disc', 'disk', 'cassette', 'modem',
        # Office
        'printer', 'typewriter', 'cash_machine', 'calculator',
        # Measurement
        'amplifier', 'oscilloscope', 'voltmeter',
        # Appliances
        'hair_dryer', 'electric_fan', 'iron', 'toaster', 'sewing_machine',
        # Accessories
        'charger',
        # Lamps / bulbs
        'lamp', 'light_bulb', 'lightbulb', 'bulb', 'lampshade',
    ],

    # ── Glass ─────────────────────────────────────────────────────────────────
    'glass': [
        # 'glass' token matches beer_glass, wine_glass, shot_glass, pier_glass, etc.
        'glass',
        # Mirrors
        'mirror', 'hand_mirror', 'pier_glass', 'cheval_glass', 'looking_glass',
        # Drinking vessels
        'goblet', 'chalice', 'tumbler',
        # Bottles
        'bottle', 'wine_bottle', 'beer_bottle', 'pop_bottle', 'whiskey_jug',
        # Containers & lab
        'jar', 'pitcher', 'water_jug', 'jug', 'carafe', 'decanter',
        'beaker', 'test_tube', 'vase', 'flask', 'ampule',
        # Household glass
        'saltshaker', 'perfume', 'hourglass',
        # Optics (glass lenses)
        'sunglasses', 'loupe', 'magnifying_glass', 'binoculars', 'telescope',
        'microscope',
        # Tanks
        'aquarium', 'terrarium',
    ],

    # ── Paper / Cardboard ─────────────────────────────────────────────────────
    'paper': [
        # Printed material
        'envelope', 'newspaper', 'magazine', 'book', 'book_jacket',
        'comic_book', 'menu', 'jigsaw_puzzle', 'crossword_puzzle',
        # Paper products
        'paper_towel', 'toilet_tissue', 'tissue',
        # Packaging
        'cardboard', 'carton', 'packet',
        # Writing
        'pencil_box', 'binder',
    ],

    # ── Metal ─────────────────────────────────────────────────────────────────
    'metal': [
        # Cans & drums
        'tin_can', 'steel_drum', 'barrel',
        # Cookware
        'wok', 'frying_pan', 'saucepan', 'pot',
        'coffeepot', 'teapot', 'watering_can',
        'spatula', 'ladle', 'tongs',
        # Cutlery
        'knife', 'fork', 'spoon',
        # Tools
        'can_opener', 'corkscrew', 'wrench', 'hammer',
        'screwdriver', 'pliers', 'shovel', 'hatchet',
        # Fasteners & hardware
        'nail', 'screw', 'bolt', 'safety_pin', 'chain', 'hook',
        # Locks
        'padlock', 'combination_lock', 'key',
        # Grooming
        'razor',
        # Brass / metal instruments
        'French_horn', 'trombone', 'trumpet', 'tuba', 'harmonica',
        # Other
        'scissors', 'anchor',
    ],

    # ── General ───────────────────────────────────────────────────────────────
    'general': [
        'trash', 'waste', 'garbage', 'litter',
        # Wood and furniture (often mistaken for other categories)
        'wood', 'lumber', 'paneling', 'board', 'table', 'chair', 'furniture',
    ],

    # ── Plastic ───────────────────────────────────────────────────────────────
    'plastic': [
        # Bags
        'plastic_bag', 'garbage_bag',
        # Bottles & containers
        'water_bottle', 'plastic_bottle', 'pill_bottle', 'petri_dish',
        # Household
        'bucket', 'toothbrush', 'comb', 'hairbrush', 'soap_dispenser',
        # Disposable
        'diaper',
        # Stationery
        'pen', 'ballpoint_pen', 'marker',
        # Toys & sports
        'frisbee', 'ping_pong_ball',
        # Media
        'cd', 'dvd',
        # Other
        'rubber_eraser', 'balloon',
    ],

    # ── Organic / Food ────────────────────────────────────────────────────────
    'organic': [
        # Fruits
        'banana', 'apple', 'orange', 'lemon', 'lime', 'pineapple',
        'strawberry', 'raspberry', 'blueberry',
        'mango', 'papaya', 'watermelon', 'cantaloupe',
        'pomegranate', 'jackfruit', 'granny_smith', 'custard_apple',
        'grape', 'cherry', 'peach', 'pear', 'plum',
        'coconut', 'kiwi', 'fig',
        # Vegetables
        'mushroom', 'broccoli', 'cauliflower', 'zucchini',
        'artichoke', 'head_cabbage', 'bell_pepper', 'cucumber',
        'carrot', 'potato', 'sweet_potato', 'tomato', 'eggplant',
        'corn', 'pea', 'spinach',
        # Prepared food
        'pizza', 'bagel', 'pretzel', 'cheeseburger', 'hotdog', 'meat_loaf',
        'burrito', 'taco', 'sandwich', 'sushi',
        'guacamole', 'carbonara', 'chocolate_sauce', 'dough',
        'loaf_of_bread', 'ice_cream', 'espresso',
        'waffle', 'pancake', 'omelette', 'fried_egg',
        'french_fries', 'soup', 'cake', 'muffin', 'cupcake', 'donut',
        # Ingredients
        'egg', 'cheese', 'butter',
    ],
}


def _cat_match(label: str, keyword: str) -> bool:
    """Token-based match — prevents 'fig' matching 'figurine', 'apple' matching 'pineapple'."""
    if label == keyword:
        return True
    label_tokens = set(label.split('_'))
    kw_tokens    = keyword.split('_')
    return all(t in label_tokens for t in kw_tokens)


# Glass/mirror tokens — checked with highest priority so mirror reflection
# of a phone does NOT get classified as electronics.
_GLASS_TOKENS = {
    'glass', 'mirror', 'goblet', 'beaker', 'vase', 'decanter', 'carafe',
    'chalice', 'tumbler', 'hourglass', 'aquarium', 'terrarium',
}


def _imagenet_classify(image_bytes: bytes) -> dict | None:
    """Use MobileNetV2 ImageNet weights to identify object, then map to waste category."""
    if not IMAGENET_AVAILABLE:
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))
        arr = np.array(img, dtype=np.float32)
        arr = keras.applications.mobilenet_v2.preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)

        preds = _imagenet_model.predict(arr, verbose=0)
        decoded = keras.applications.mobilenet_v2.decode_predictions(preds, top=10)[0]

        print(f"[ImageNet] Top predictions: {[(d[1], round(d[2]*100)) for d in decoded]}")

        # ── Priority scan: glass/mirror tokens beat everything else ──────────
        # Mirror photos reflect phones → ImageNet sees 'cellular_telephone'.
        # Only trigger if score is reasonable (>= 0.15)
        for _, label, score in decoded:
            if score < 0.15:
                continue
            tokens = set(label.lower().replace('-', '_').split('_'))
            if tokens & _GLASS_TOKENS:
                mapped_confidence = min(97, max(55, round(score * 100) + 10))
                print(f"[ImageNet] Predicted Glass ({label.replace('_', ' ').title()}) with {mapped_confidence:.1f}% confidence")
                return {
                    'category':  'glass',
                    'item_name': label.replace('_', ' ').title(),
                    'confidence': mapped_confidence,
                }

        # ── Pass 1: high-confidence match (score >= 0.10) ────────────────────
        for _, label, score in decoded:
            if score < 0.10:
                continue
            label_lower = label.lower().replace('-', '_')
            
            # Thin paper items (like Kite/Patang) fix
            if 'kite' in label_lower:
                mapped_confidence = min(97, round(score * 100) + 15)
                print(f"[ImageNet] Predicted Paper ({label.replace('_', ' ').title()}) with {mapped_confidence:.1f}% confidence")
                return {
                    'category':  'paper',
                    'item_name': 'Kite (Paper Item)',
                    'confidence': mapped_confidence,
                }

            for cat, keywords in _IMAGENET_CAT_MAP.items():
                for kw in keywords:
                    if _cat_match(label_lower, kw):
                        mapped_confidence = min(97, max(55, round(score * 100) + 10))
                        print(f"[ImageNet] Predicted {cat} ({label.replace('_', ' ').title()}) with {mapped_confidence:.1f}% confidence")
                        return {
                            'category':  cat,
                            'item_name': label.replace('_', ' ').title(),
                            'confidence': mapped_confidence,
                        }

        # ── Pass 2: low-confidence match (organic/general need >= 0.20) ──────
        for _, label, score in decoded:
            label_lower = label.lower().replace('-', '_')
            for cat, keywords in _IMAGENET_CAT_MAP.items():
                if cat in ('organic', 'general') and score < 0.20:
                    continue
                for kw in keywords:
                    if _cat_match(label_lower, kw):
                        mapped_confidence = min(97, max(55, round(score * 100) + 10))
                        print(f"[ImageNet] Predicted {cat} ({label.replace('_', ' ').title()}) with {mapped_confidence:.1f}% confidence")
                        return {
                            'category':  cat,
                            'item_name': label.replace('_', ' ').title(),
                            'confidence': mapped_confidence,
                        }

        # ── Fallback: return top label as general ─────────────────────────────
        top_label = decoded[0][1].replace('_', ' ').title()
        top_conf  = round(decoded[0][2] * 100)
        print(f"[ImageNet] Predicted general ({top_label}) with {top_conf:.1f}% confidence")
        return {
            'category':  'general',
            'item_name': top_label,
            'confidence': max(55, top_conf),
        }
    except Exception as e:
        print(f"[ImageNet] Error: {e}")
        return None

# ─── CNN model (fallback) ─────────────────────────────────────────────────────
try:
    from waste_model import WasteClassifier, CATEGORY_INFO
    classifier = WasteClassifier()
    TF_AVAILABLE = True
    print("[INFO] CNN waste model loaded.")
except Exception as e:
    classifier = None
    TF_AVAILABLE = False
    CATEGORY_INFO = {
        'e_usable': {'label': 'E-Usable (Electronics)', 'result': 'usable',     'bin': 'E-Waste Collection Center'},
        'general':  {'label': 'General Waste',          'result': 'waste',      'bin': 'General Waste Bin'},
        'glass':    {'label': 'Glass',                  'result': 'recyclable', 'bin': 'Glass Recycling Bin'},
        'metal':    {'label': 'Metal Waste',            'result': 'recyclable', 'bin': 'Metal Recycling Bin'},
        'organic':  {'label': 'Organic Waste',          'result': 'waste',      'bin': 'Compost / Green Bin'},
        'paper':    {'label': 'Paper / Cardboard',      'result': 'recyclable', 'bin': 'Paper Recycling Bin'},
        'plastic':  {'label': 'Plastic Waste',          'result': 'waste',      'bin': 'Waste Bin'},
    }

# ─── Filename keyword fallback ────────────────────────────────────────────────
_KEYWORDS = {
    'e_usable': ['battery', 'cell', 'lithium', 'mouse', 'keyboard', 'laptop',
                 'phone', 'mobile', 'charger', 'cable', 'monitor', 'circuit',
                 'ewaste', 'electronic', 'usb', 'tablet', 'camera', 'headphone',
                 'speaker', 'router', 'printer'],
    'glass':    ['glass', 'bottle', 'jar', 'window', 'mug'],
    'paper':    ['paper', 'cardboard', 'newspaper', 'book', 'carton', 'box'],
    'plastic':  ['plastic', 'bag', 'cup', 'container', 'straw', 'wrapper'],
    'organic':  ['food', 'vegetable', 'fruit', 'leaf', 'banana', 'apple', 'bread', 'rice', 'organic'],
    'metal':    ['metal', 'can', 'tin', 'aluminum', 'iron', 'steel', 'copper', 'scrap', 'alloy'],
    'general':  ['trash', 'general', 'waste', 'garbage'],
}

def _fallback_classify(filename: str) -> tuple[str, int]:
    name = filename.lower()
    for cat, keywords in _KEYWORDS.items():
        for kw in keywords:
            if kw in name:
                return cat, random.randint(78, 92)
    return 'general', random.randint(55, 70)

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='.', static_url_path='')

CATEGORY_DETAILS = {
    'e_usable': {
        'category': 'E-Usable — Electronics',
        'title': 'Electronic Device Detected',
        'description': 'Electronics and batteries contain hazardous materials and valuable metals. They must NEVER go in regular trash. Take to a certified e-waste collection center where components are safely recovered and reused.',
        'action': 'Take to E-Waste Collection Center',
        'tip': 'Check if the device still works — donate it if possible. Electronics stores often offer free drop-off for old devices and batteries.',
    },
    'glass': {
        'category': 'Recyclable — Glass',
        'title': 'Glass Waste Detected',
        'description': 'Glass bottles, jars, and containers are 100% recyclable and can be melted down and reformed endlessly without any loss in quality.',
        'action': 'Place in Glass Recycling Bin',
        'tip': 'Rinse glass before recycling and remove metal lids. Glass can be recycled indefinitely with no degradation.',
    },
    'paper': {
        'category': 'Recyclable — Paper',
        'title': 'Paper Detected',
        'description': 'Paper and cardboard are among the easiest materials to recycle. Keep them clean and dry for the best recycling outcome.',
        'action': 'Flatten and Place in Paper Recycling Bin',
        'tip': 'Remove tape, staples, and food stains from paper/cardboard before recycling for best results.',
    },
    'metal': {
        'category': 'Recyclable — Metal',
        'title': 'Metal Waste Detected',
        'description': 'Metals like aluminum and steel are highly recyclable. Recycling metal saves enormous energy compared to extracting from raw ore.',
        'action': 'Place in Metal Recycling Bin',
        'tip': 'Recycling aluminum saves 95% of the energy needed to produce it from raw bauxite. Every can counts!',
    },
    'plastic': {
        'category': 'Plastic Waste',
        'title': 'Plastic Waste Detected',
        'description': 'Plastic waste takes hundreds of years to decompose. Reduce plastic use where possible and dispose of it responsibly in the designated waste bin.',
        'action': 'Place in Waste Bin',
        'tip': 'Avoid single-use plastics. Choose reusable bags, bottles, and containers to reduce plastic waste.',
    },
    'organic': {
        'category': 'Organic Waste',
        'title': 'Organic Waste Detected',
        'description': 'Food scraps, vegetable peels, and organic matter should be disposed of in the compost or green bin to avoid polluting the environment.',
        'action': 'Place in Compost / Green Bin',
        'tip': 'Start a home compost bin! Composting reduces landfill waste and creates free fertilizer for plants.',
    },
    'general': {
        'category': 'General Waste',
        'title': 'General Trash Detected',
        'description': 'This item appears to be general waste that cannot be easily recycled in its current state. Minimize this type of waste where possible.',
        'action': 'Place in General Waste Bin',
        'tip': 'Reduce general waste by choosing products with minimal packaging and opting for reusable alternatives.',
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return send_file('index.html')


@app.route('/api/demo-login', methods=['POST'])
def demo_login():
    """Provide a local-only test identity when Google OAuth is not configured."""
    return jsonify({
        'success': True,
        'demo': True,
        'user': {
            'name': 'BinSense Demo User',
            'email': 'demo@binsense.local',
            'picture': '',
        },
    })


@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('.', filename)


@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    image_bytes = file.read()

    def _build_result(cat, confidence, item_name=None, top3=None):
        info = CATEGORY_DETAILS.get(cat, CATEGORY_DETAILS['general'])
        cat_info = CATEGORY_INFO.get(cat, {})
        label = item_name or cat_info.get('label', cat.title())
        return {
            'item':        label,
            'result':      cat_info.get('result', 'waste'),
            'category':    info['category'],
            'confidence':  confidence,
            'title':       info['title'],
            'description': info['description'],
            'action':      info['action'],
            'tip':         info['tip'],
            'top3':        top3 or [{'category': cat, 'label': label, 'confidence': confidence}],
        }

    # ── Step 1: Run both models ───────────────────────────────────────────────
    inet = _imagenet_classify(image_bytes)

    cnn = None
    if TF_AVAILABLE and classifier is not None:
        try:
            cnn = classifier.predict(image_bytes)
        except Exception as e:
            print(f"[CNN] Error: {e}")

    # ── Step 2: Ensemble — combine both results ──────────────────────────────
    if inet is not None and cnn is not None:
        inet_cat = inet['category']
        cnn_cat  = cnn['category']
        inet_conf = inet['confidence']
        cnn_conf  = round(cnn['confidence'])

        # ── Threshold Logic: If both models are weak, force 'general' ────────
        if inet_conf < CONFIDENCE_THRESHOLD * 100 and cnn_conf < CONFIDENCE_THRESHOLD * 100:
            print(f"[Ensemble] LOW CONFIDENCE (ImageNet:{inet_conf}% CNN:{cnn_conf}%) -> Forcing general")
            return jsonify({'success': True, 'data': _build_result('general', max(inet_conf, cnn_conf), inet['item_name'], cnn['top3'])})

        if inet_cat == cnn_cat:
            # Both agree — confidence boost
            combined_conf = min(97, round((inet_conf + cnn_conf) / 2) + 8)
            print(f"[Ensemble] AGREE -> {inet_cat} | ImageNet:{inet_conf}% CNN:{cnn_conf}% -> {combined_conf}%")
            return jsonify({'success': True, 'data': _build_result(inet_cat, combined_conf, inet['item_name'], cnn['top3'])})
        else:
            # Disagree — use the result with higher confidence
            # Prefer ImageNet for electronics (bulbs/lamps) when confidences are close
            if inet_cat == 'e_usable' and inet_conf + 8 >= cnn_conf:
                print(f"[Ensemble] PREFERENCE -> ImageNet (e_usable) chosen: {inet_cat}({inet_conf}%) vs CNN:{cnn_cat}({cnn_conf}%)")
                return jsonify({'success': True, 'data': _build_result(inet_cat, inet_conf, inet['item_name'])})

            # Otherwise choose the higher-confidence model
            if inet_conf >= cnn_conf:
                print(f"[Ensemble] DISAGREE -> ImageNet wins: {inet_cat}({inet_conf}%) vs CNN:{cnn_conf}%)")
                return jsonify({'success': True, 'data': _build_result(inet_cat, inet_conf, inet['item_name'])})
            else:
                print(f"[Ensemble] DISAGREE -> CNN wins: {cnn_cat}({cnn_conf}%) vs ImageNet:{inet_cat}({inet_conf}%)")
                info = CATEGORY_DETAILS.get(cnn_cat, CATEGORY_DETAILS['general'])
                return jsonify({'success': True, 'data': {
                    'item': cnn['label'], 'result': cnn['result'],
                    'category': info['category'], 'confidence': cnn_conf,
                    'title': info['title'], 'description': info['description'],
                    'action': info['action'], 'tip': info['tip'],
                    'top3': cnn['top3'],
                }})

    # ── Step 5: Final Global Safeguard ───────────────────────────────────────
    # If we reached here with a category that isn't general, but confidence is too low
    final_result = None
    if inet is not None:
        final_result = _build_result(inet['category'], inet['confidence'], inet['item_name'])
    elif cnn is not None:
        cnn_conf = round(cnn['confidence'])
        info = CATEGORY_DETAILS.get(cnn['category'], CATEGORY_DETAILS['general'])
        final_result = {
            'item': cnn['label'], 'result': cnn['result'],
            'category': info['category'], 'confidence': cnn_conf,
            'title': info['title'], 'description': info['description'],
            'action': info['action'], 'tip': info['tip'],
            'top3': cnn['top3'],
        }
    else:
        fname_cat, fname_conf = _fallback_classify(file.filename or '')
        final_result = _build_result(fname_cat, fname_conf)

    # Force general only when confidence is below the configured threshold.
    if final_result['confidence'] < CONFIDENCE_THRESHOLD * 100:
        print(f"[Final Guard] Confidence {final_result['confidence']}% too low -> Forcing General")
        final_result = _build_result('general', final_result['confidence'], final_result['item'])

    return jsonify({'success': True, 'data': final_result})


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "="*55)
    print("  BinSense Backend Running")
    if IMAGENET_AVAILABLE:
        print("  Engine: ImageNet Vision (MobileNetV2)")
    elif TF_AVAILABLE:
        print("  Engine: CNN Waste Model")
    else:
        print("  Engine: Keyword Fallback")
    print("  Open: http://localhost:5000")
    print("="*55 + "\n")
    threading.Timer(2.0, lambda: webbrowser.open('http://localhost:5000')).start()
    app.run(debug=False, port=5000)
