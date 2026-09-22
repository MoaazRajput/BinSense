
import unittest
import io
from app import app

class TestBinSenseAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home_route(self):
        """Check if home page loads"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_analyze_no_image(self):
        """Check error if no image is sent"""
        response = self.app.post('/api/analyze')
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'No image provided', response.data)

    def test_analyze_dummy_image(self):
        """Test with a dummy image file"""
        data = {
            'image': (io.BytesIO(b"abcdef"), 'test.jpg')
        }
        response = self.app.post('/api/analyze', data=data, content_type='multipart/form-data')
        # Since it's a dummy image, it should fallback to keyword search or general
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertTrue(json_data['success'])
        self.assertIn('category', json_data['data'])

if __name__ == '__main__':
    unittest.main()
