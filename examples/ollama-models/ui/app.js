// Ollama Chat - GPU CLI Web UI
// Lightweight chat interface for Ollama models

const OLLAMA_API = 'http://localhost:11434';
const MODELS_CONFIG = './models.json';
const MAX_HISTORY_LENGTH = 50;  // Limit conversation history to prevent memory issues

// State
let currentModel = null;
let conversationHistory = [];
let isStreaming = false;

// Trim conversation history to prevent unbounded memory growth
function trimHistory() {
  if (conversationHistory.length > MAX_HISTORY_LENGTH) {
    // Keep system message if present, then most recent messages
    const hasSystem = conversationHistory[0]?.role === 'system';
    const startIndex = hasSystem ? 1 : 0;
    const keepCount = MAX_HISTORY_LENGTH - startIndex;
    conversationHistory = hasSystem
      ? [conversationHistory[0], ...conversationHistory.slice(-keepCount)]
      : conversationHistory.slice(-keepCount);
  }
}

// DOM Elements
const modelSelect = document.getElementById('model-select');
const pullModelBtn = document.getElementById('pull-model-btn');
const chatContainer = document.getElementById('chat-container');
const messagesDiv = document.getElementById('messages');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');

// Modal elements
const pullModal = document.getElementById('pull-modal');
const modalBackdrop = document.getElementById('modal-backdrop');
const modelNameInput = document.getElementById('model-name-input');
const pullConfirmBtn = document.getElementById('pull-confirm');
const pullCancelBtn = document.getElementById('pull-cancel');
const pullProgress = document.getElementById('pull-progress');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
  await checkConnection();
  await loadModels();
  setupEventListeners();
  autoResizeTextarea();
});

// Check Ollama connection
async function checkConnection() {
  try {
    const response = await fetch(`${OLLAMA_API}/api/tags`);
    if (response.ok) {
      setStatus('connected', 'Connected');
      return true;
    }
  } catch (e) {
    setStatus('disconnected', 'Disconnected');
  }
  return false;
}

// Set status indicator
function setStatus(state, text) {
  statusDot.className = `status-dot ${state}`;
  statusText.textContent = text;
}

// Load available models
async function loadModels() {
  try {
    // First, try to get configured models
    let configuredModels = [];
    let defaultModel = null;
    try {
      const configResponse = await fetch(MODELS_CONFIG);
      if (configResponse.ok) {
        const config = await configResponse.json();
        configuredModels = config.models || [];
        defaultModel = config.default;
      }
    } catch (e) {
      console.log('No models.json found, using available models');
    }

    // Get actually available models from Ollama
    const response = await fetch(`${OLLAMA_API}/api/tags`);
    if (!response.ok) throw new Error('Failed to fetch models');

    const data = await response.json();
    const availableModels = data.models || [];

    // Clear and populate select
    modelSelect.innerHTML = '';

    if (availableModels.length === 0) {
      modelSelect.innerHTML = '<option value="">No models available - pull one first</option>';
      sendBtn.disabled = true;
      return;
    }

    // Add available models to select
    availableModels.forEach(model => {
      const option = document.createElement('option');
      option.value = model.name;
      option.textContent = `${model.name} (${formatSize(model.size)})`;
      modelSelect.appendChild(option);
    });

    // Select default model if available
    if (defaultModel && availableModels.some(m => m.name === defaultModel || m.name.startsWith(defaultModel))) {
      const match = availableModels.find(m => m.name === defaultModel || m.name.startsWith(defaultModel));
      if (match) modelSelect.value = match.name;
    }

    currentModel = modelSelect.value;
    sendBtn.disabled = !currentModel;

  } catch (e) {
    console.error('Error loading models:', e);
    modelSelect.innerHTML = '<option value="">Error loading models</option>';
    setStatus('error', 'Error');
  }
}

// Format file size
function formatSize(bytes) {
  if (!bytes) return 'unknown';
  const gb = bytes / (1024 * 1024 * 1024);
  return gb >= 1 ? `${gb.toFixed(1)} GB` : `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
}

// Setup event listeners
function setupEventListeners() {
  // Model selection
  modelSelect.addEventListener('change', (e) => {
    currentModel = e.target.value;
    sendBtn.disabled = !currentModel;
  });

  // Chat form submit
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    await sendMessage();
  });

  // Enter to send (Shift+Enter for newline)
  userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event('submit'));
    }
  });

  // Pull model button
  pullModelBtn.addEventListener('click', () => showPullModal());
  pullCancelBtn.addEventListener('click', () => hidePullModal());
  modalBackdrop.addEventListener('click', () => hidePullModal());
  pullConfirmBtn.addEventListener('click', () => pullModel());

  // Enter to confirm pull
  modelNameInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      pullModel();
    }
  });
}

// Auto-resize textarea
function autoResizeTextarea() {
  userInput.addEventListener('input', () => {
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 200) + 'px';
  });
}

// Send message
async function sendMessage() {
  const message = userInput.value.trim();
  if (!message || !currentModel || isStreaming) return;

  // Clear welcome message if present
  const welcome = messagesDiv.querySelector('.welcome-message');
  if (welcome) welcome.remove();

  // Add user message
  addMessage('user', message);
  userInput.value = '';
  userInput.style.height = 'auto';

  // Add to history
  conversationHistory.push({ role: 'user', content: message });

  // Show loading state
  isStreaming = true;
  sendBtn.disabled = true;
  sendBtn.querySelector('.btn-text').classList.add('hidden');
  sendBtn.querySelector('.btn-loading').classList.remove('hidden');

  // Create assistant message placeholder
  const assistantDiv = addMessage('assistant', '');
  const contentDiv = assistantDiv.querySelector('.message-content');

  try {
    // Stream response
    const response = await fetch(`${OLLAMA_API}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: currentModel,
        messages: conversationHistory,
        stream: true
      })
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let fullResponse = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n').filter(line => line.trim());

      for (const line of lines) {
        try {
          const data = JSON.parse(line);
          if (data.message?.content) {
            fullResponse += data.message.content;
            contentDiv.innerHTML = formatMarkdown(fullResponse);
            scrollToBottom();
          }
        } catch (e) {
          // Log non-JSON lines in debug mode (incomplete chunks are expected during streaming)
          if (line.trim() && !line.includes('"done"')) {
            console.debug('Skipped non-JSON chunk:', line.substring(0, 100));
          }
        }
      }
    }

    // Add to history and trim if needed
    conversationHistory.push({ role: 'assistant', content: fullResponse });
    trimHistory();

  } catch (e) {
    console.error('Error sending message:', e);
    contentDiv.innerHTML = `<span class="error">Error: ${escapeHtml(e.message)}</span>`;
  } finally {
    isStreaming = false;
    sendBtn.disabled = false;
    sendBtn.querySelector('.btn-text').classList.remove('hidden');
    sendBtn.querySelector('.btn-loading').classList.add('hidden');
  }
}

// Add message to chat
function addMessage(role, content) {
  const div = document.createElement('div');
  div.className = `message ${role}`;
  div.innerHTML = `
    <div class="message-avatar">${role === 'user' ? 'You' : 'AI'}</div>
    <div class="message-content">${content ? formatMarkdown(content) : '<span class="typing">...</span>'}</div>
  `;
  messagesDiv.appendChild(div);
  scrollToBottom();
  return div;
}

// Escape HTML to prevent XSS attacks
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// Basic markdown formatting (escapes HTML first to prevent XSS)
function formatMarkdown(text) {
  const escaped = escapeHtml(text);
  return escaped
    .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>')
    .replace(/\n/g, '<br>');
}

// Scroll to bottom of chat
function scrollToBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Show pull modal
function showPullModal() {
  pullModal.classList.remove('hidden');
  modelNameInput.value = '';
  modelNameInput.focus();
  pullProgress.classList.add('hidden');
}

// Hide pull modal
function hidePullModal() {
  pullModal.classList.add('hidden');
  modelNameInput.value = '';
  pullProgress.classList.add('hidden');
}

// Pull model
async function pullModel() {
  const modelName = modelNameInput.value.trim();
  if (!modelName) return;

  pullConfirmBtn.disabled = true;
  pullProgress.classList.remove('hidden');
  progressText.textContent = 'Starting download...';
  progressFill.style.width = '0%';

  try {
    const response = await fetch(`${OLLAMA_API}/api/pull`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: modelName, stream: true })
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n').filter(line => line.trim());

      for (const line of lines) {
        try {
          const data = JSON.parse(line);
          if (data.status) {
            progressText.textContent = data.status;
          }
          if (data.completed && data.total) {
            const percent = (data.completed / data.total) * 100;
            progressFill.style.width = `${percent}%`;
          }
        } catch (e) {
          // Ignore parse errors
        }
      }
    }

    progressText.textContent = 'Model pulled successfully!';
    progressFill.style.width = '100%';

    // Reload models after short delay
    setTimeout(async () => {
      await loadModels();
      hidePullModal();
    }, 1000);

  } catch (e) {
    console.error('Error pulling model:', e);
    progressText.textContent = `Error: ${e.message}`;
  } finally {
    pullConfirmBtn.disabled = false;
  }
}
