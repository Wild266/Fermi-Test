const heroMetrics = {
  questions: document.querySelector('[data-metric="questions"]'),
  'accuracy-base': document.querySelector('[data-metric="accuracy-base"]'),
  'accuracy-reasoning': document.querySelector('[data-metric="accuracy-reasoning"]')
};

const questionListEl = document.getElementById('question-list');
const searchInput = document.getElementById('search');
const filterImproved = document.getElementById('filter-improved');
const modal = document.getElementById('question-modal');
const modalClose = modal?.querySelector('.modal__close');
const modalQuestion = document.getElementById('modal-question');
const modalMeta = document.getElementById('modal-meta');
const modalBaseSummary = document.getElementById('modal-base-summary');
const modalReasoningSummary = document.getElementById('modal-reasoning-summary');
const modalBasePlot = document.getElementById('modal-base-plot');
const modalReasoningPlot = document.getElementById('modal-reasoning-plot');
const footerYear = document.getElementById('year');

if (footerYear) {
  footerYear.textContent = new Date().getFullYear();
}

let questions = [];
let filtered = [];

fetch('data/questions.json')
  .then((res) => res.json())
  .then((data) => {
    questions = data;
    updateMetrics(data);
    filtered = [...questions];
    renderQuestions();
  })
  .catch((err) => {
    console.error('Failed to load question data', err);
  });

function updateMetrics(data) {
  const total = data.length;
  const baseExact = data.filter((item) => item.base_correct).length;
  const reasoningExact = data.filter((item) => item.reasoning_correct).length;

  if (heroMetrics.questions) {
    heroMetrics.questions.textContent = total.toString();
  }
  if (heroMetrics['accuracy-base']) {
    heroMetrics['accuracy-base'].textContent = `${percentage(baseExact, total)} exact`;
  }
  if (heroMetrics['accuracy-reasoning']) {
    heroMetrics['accuracy-reasoning'].textContent = `${percentage(reasoningExact, total)} exact`;
  }
}

function percentage(value, total) {
  if (!total) return '0%';
  return `${(value / total * 100).toFixed(1)}%`;
}

function renderQuestions() {
  if (!questionListEl) return;
  questionListEl.innerHTML = '';
  const frag = document.createDocumentFragment();
  filtered.forEach((item) => {
    const card = document.createElement('article');
    card.className = 'question-card';
    card.setAttribute('role', 'listitem');
    card.tabIndex = 0;

    const id = document.createElement('div');
    id.className = 'question-card__id';
    id.textContent = `#${item.slug}`;

    const question = document.createElement('h4');
    question.className = 'question-card__question';
    question.textContent = item.question;

    const badges = document.createElement('div');
    badges.className = 'question-card__badges';

    badges.appendChild(makeBadge(`GT 10^${item.correct_answer_log10}`));

    const baseLabel = item.base_correct ? 'Baseline exact' : item.base_within_10 ? 'Baseline ±1' : 'Baseline miss';
    const baseBadge = makeBadge(baseLabel, item.base_correct || item.base_within_10);
    badges.appendChild(baseBadge);

    const reasoningLabel = item.reasoning_correct ? 'Reasoning exact' : item.reasoning_within_10 ? 'Reasoning ±1' : 'Reasoning miss';
    const reasoningBadge = makeBadge(reasoningLabel, item.reasoning_correct || item.reasoning_within_10);
    badges.appendChild(reasoningBadge);

    card.append(id, question, badges);
    card.addEventListener('click', () => openModal(item));
    card.addEventListener('keypress', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        openModal(item);
      }
    });

    frag.appendChild(card);
  });
  questionListEl.appendChild(frag);
}

function makeBadge(label, positive = false) {
  const badge = document.createElement('span');
  badge.className = 'badge';
  if (positive) badge.classList.add('badge--success');
  if (!positive && /miss/i.test(label)) badge.classList.add('badge--warning');
  badge.textContent = label;
  return badge;
}

function openModal(item) {
  if (!modal) return;
  modalQuestion.textContent = item.question;
  modalMeta.textContent = `Correct log₁₀: ${item.correct_answer_log10} · Baseline: ${item.base_answer_log10 || '–'} · Reasoning: ${item.reasoning_answer_log10 || '–'}`;
  modalBaseSummary.textContent = item.base_summary || 'No baseline transcript available.';
  modalReasoningSummary.textContent = item.reasoning_summary || 'No reasoning transcript available.';
  modalBasePlot.src = item.plots.base;
  modalReasoningPlot.src = item.plots.reasoning;
  modal.showModal();
}

modalClose?.addEventListener('click', () => modal.close());
modal?.addEventListener('close', () => {
  modalBasePlot.removeAttribute('src');
  modalReasoningPlot.removeAttribute('src');
});

[searchInput, filterImproved].forEach((control) => {
  control?.addEventListener('input', applyFilters);
  control?.addEventListener('change', applyFilters);
});

function applyFilters() {
  const query = (searchInput?.value || '').trim().toLowerCase();
  const improvedOnly = filterImproved?.checked;
  filtered = questions.filter((item) => {
    const matchesQuery = query
      ? item.question.toLowerCase().includes(query)
        || item.base_summary.toLowerCase().includes(query)
        || item.reasoning_summary.toLowerCase().includes(query)
      : true;

    const improved = (item.reasoning_correct && !item.base_correct)
      || (!item.base_correct && item.reasoning_within_10 && !item.base_within_10)
      || (item.reasoning_within_10 && !item.base_within_10);

    const matchesImprovement = improvedOnly ? improved : true;
    return matchesQuery && matchesImprovement;
  });
  renderQuestions();
}

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && modal?.open) {
    modal.close();
  }
});