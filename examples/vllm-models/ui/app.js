// vLLM Chat - GPU CLI Web UI
// Lightweight chat interface for vLLM OpenAI-compatible API

const VLLM_API = 'http://localhost:8000';
const MODELS_CONFIG = './models.json';
const MAX_HISTORY_LENGTH = 50;

// State
let currentModel = null;
let conversationHistory = [];
let isStreaming = false;

// Trim conversation history to prevent unbounded memory growth
function trimHistory() {
  if (conversationHistory.length > MAX_HISTORY_LENGTH) {
    const hasSystem = conversationHistory[0]?.role === 'system';
    const startIndex = hasSystem ? 1 : 0;
    const keepCount = MAX_HISTORY_LENGTH - startIndex;
    conversationHistory = hasSystem
      ? [conversationHistory[0], ...conversationHistory.slice(-keepCount)]
      : conversationHistory.slice(-keepCount);
  }
}

// DOM Elements
const modelName = document.getElementById('model-name');
const chatContainer = document.getElementById('chat-container');
const messagesDiv = document.getElementById('messages');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
  await loadModelConfig();
  await checkConnection();
  setupEventListeners();
  autoResizeTextarea();
});

// Load model configuration from models.json
async function loadModelConfig() {
  try {
    const response = await fetch(MODELS_CONFIG);
    if (response.ok) {
      const config = await response.json();
      currentModel = config.model;
      modelName.textContent = formatModelName(currentModel);
      modelName.title = currentModel; // Full name on hover
    }
  } catch (e) {
    console.log('Could not load models.json, will use API to detect model');
  }
}

// Format model name for display (extract short name from HF path)
function formatModelName(fullName) {
  if (!fullName) return 'Unknown';
  const parts = fullName.split('/');
  return parts[parts.length - 1];
}

// Check vLLM connection and get model info
async function checkConnection() {
  try {
    const response = await fetch(`${VLLM_API}/v1/models`);
    if (response.ok) {
      const data = await response.json();
      if (data.data && data.data.length > 0) {
        // Use the model from API if not set from config
        if (!currentModel) {
          currentModel = data.data[0].id;
          modelName.textContent = formatModelName(currentModel);
          modelName.title = currentModel;
        }
        setStatus('connected', 'Connected');
        sendBtn.disabled = false;
        return true;
      }
    }
  } catch (e) {
    console.error('Connection error:', e);
  }
  setStatus('disconnected', 'Disconnected');
  sendBtn.disabled = true;
  return false;
}

// Set status indicator
function setStatus(state, text) {
  statusDot.className = `status-dot ${state}`;
  statusText.textContent = text;
}

// Setup event listeners
function setupEventListeners() {
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
    // Stream response using OpenAI-compatible API
    const response = await fetch(`${VLLM_API}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: currentModel,
        messages: conversationHistory,
        stream: true
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${errorText}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let fullResponse = '';
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Process SSE format: data: {json}\n\n
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line in buffer

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith('data: ')) continue;

        const data = trimmed.slice(6); // Remove 'data: ' prefix

        // Check for stream end
        if (data === '[DONE]') continue;

        try {
          const parsed = JSON.parse(data);
          const delta = parsed.choices?.[0]?.delta?.content;
          if (delta) {
            fullResponse += delta;
            contentDiv.innerHTML = formatMarkdown(fullResponse);
            scrollToBottom();
          }
        } catch (e) {
          // Skip malformed JSON chunks
          console.debug('Skipped chunk:', data.substring(0, 50));
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
