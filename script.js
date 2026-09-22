/* BinSense — main JS */

/* ---- Navbar: add shadow on scroll ---- */
const nav = document.querySelector('#mainNav');
window.addEventListener('scroll', () => {
  nav?.classList.toggle('scrolled', window.scrollY > 40);
});

/* ---- Auth state ---- */
const syncAuthUI = () => {
  const authLink = document.getElementById('authLink');
  if (!authLink) return;

  try {
    const user = JSON.parse(localStorage.getItem('binsense_user') || 'null');

    if (user) {
      const firstName = (user.name || 'User').split(' ')[0];
      authLink.href = '#';
      authLink.className = 'btn btn-outline-green ms-3 px-4';
      authLink.innerHTML = user.picture
        ? `<img src="${user.picture}" alt="${firstName}" style="width:24px;height:24px;border-radius:50%;object-fit:cover;" class="me-2">${firstName}`
        : `<i class="bi bi-person-check-fill me-1"></i>${firstName}`;
      authLink.onclick = (event) => {
        event.preventDefault();
        localStorage.removeItem('binsense_user');
        syncAuthUI();
        showToast('Logged out successfully.');
      };
      return;
    }
  } catch (error) {
    console.warn('Auth state not available', error);
  }

  authLink.href = 'login.html';
  authLink.className = 'btn btn-green ms-3 px-4';
  authLink.innerHTML = '<i class="bi bi-person-fill me-1"></i>Login';
  authLink.onclick = null;
};

syncAuthUI();

/* ---- Scroll fade-in animations ---- */
const fadeObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
      fadeObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.12 });

document.querySelectorAll('[data-fade]').forEach(el => fadeObserver.observe(el));


/* ---- Waste Analyzer ---- */
const analyzeImage = async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  document.getElementById('analyzerResult').classList.remove('d-none');
  document.getElementById('uploadArea').classList.add('d-none');

  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById('previewImg').src = e.target.result;
  };
  reader.readAsDataURL(file);

  document.getElementById('loadingState').classList.remove('d-none');
  document.getElementById('resultState').classList.add('d-none');

  const formData = new FormData();
  formData.append('image', file);

  try {
    const response = await fetch('/api/analyze', { method: 'POST', body: formData });
    const json = await response.json();

    document.getElementById('loadingState').classList.add('d-none');

    if (json.success && json.data) {
      showResult(json.data);
    } else {
      throw new Error(json.error || 'Analysis failed');
    }
  } catch {
    document.getElementById('loadingState').classList.add('d-none');
    showToast('Could not analyze image. Make sure the server is running.');
    resetAnalyzer();
  }
};

const showResult = (data) => {
  document.getElementById('loadingState').classList.add('d-none');
  document.getElementById('resultState').classList.remove('d-none');

  const badge = document.getElementById('resultBadge');
  if (data.result === 'usable') {
    badge.className = 'result-badge badge-usable';
    badge.innerHTML = '<i class="bi bi-check-circle-fill me-1"></i>Still Usable';
  } else if (data.result === 'recyclable') {
    badge.className = 'result-badge badge-recyclable';
    badge.innerHTML = '<i class="bi bi-arrow-repeat me-1"></i>Recyclable';
  } else {
    badge.className = 'result-badge badge-waste';
    badge.innerHTML = '<i class="bi bi-exclamation-triangle-fill me-1"></i>Waste';
  }

  document.getElementById('resultTitle').textContent    = data.title;
  document.getElementById('resultDesc').textContent     = data.description;
  document.getElementById('resultCategory').textContent = data.category;
  document.getElementById('resultAction').textContent   = data.action;
  document.getElementById('resultTip').textContent      = data.tip;

  const bar = document.getElementById('confidenceBar');
  document.getElementById('confidenceText').textContent = `${data.confidence}%`;
  bar.style.width = '0%';
  requestAnimationFrame(() => {
    bar.style.transition = 'width 1s ease';
    bar.style.width = `${data.confidence}%`;
  });
};

const resetAnalyzer = () => {
  document.getElementById('analyzerResult').classList.add('d-none');
  document.getElementById('uploadArea').classList.remove('d-none');
  document.getElementById('resultState').classList.add('d-none');
  document.getElementById('loadingState').classList.add('d-none');
  document.getElementById('imageInput').value = '';
};


/* ---- Contact form ---- */
const submitForm = (event) => {
  event.preventDefault();
  showToast('Message sent! We will get back to you soon.');
  event.target.reset();
};


/* ---- Feedback widget ---- */
let selectedRating = 0;
const ratingLabels = ['', 'Poor', 'Fair', 'Good', 'Great', 'Excellent!'];

document.querySelectorAll('.star').forEach(star => {
  star.addEventListener('mouseenter', () => highlightStars(+star.dataset.val));
  star.addEventListener('mouseleave', () => highlightStars(selectedRating));
  star.addEventListener('click', () => {
    selectedRating = +star.dataset.val;
    highlightStars(selectedRating);
    document.getElementById('ratingLabel').textContent = ratingLabels[selectedRating];
    document.getElementById('ratingLabel').style.color = 'var(--green)';
  });
});

function highlightStars(count) {
  document.querySelectorAll('.star').forEach((s, i) => {
    s.classList.toggle('active', i < count);
  });
}

function submitFeedback() {
  if (!selectedRating) {
    showToast('Please select a star rating first.');
    return;
  }
  showToast(`Thanks for the ${ratingLabels[selectedRating]} rating!`);
  selectedRating = 0;
  highlightStars(0);
  document.getElementById('ratingLabel').textContent = 'Tap to rate';
  document.getElementById('ratingLabel').style.color = '';
  document.getElementById('feedbackText').value = '';
}


/* ---- Toast ---- */
const showToast = (message) => {
  const toast = document.getElementById('toast');
  document.getElementById('toastMsg').textContent = message;
  toast.classList.remove('d-none');
  setTimeout(() => toast.classList.add('d-none'), 3200);
};
