// Ollama Chat - GPU CLI Web UI
// Lightweight chat interface for Ollama models with conversation persistence

const OLLAMA_API = 'http://localhost:11434';
const MODELS_CONFIG = './models.json';
const MAX_HISTORY_LENGTH = 50;  // Limit conversation history to prevent memory issues

// SVG Avatar Icons
const USER_AVATAR = `<svg viewBox="0 0 24 24" fill="currentColor">
  <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
</svg>`;

const AI_AVATAR = `<svg viewBox="0 0 24 24" fill="currentColor">
  <path d="M12 2L9.19 8.63 2 9.24l5.46 4.73L5.82 21 12 17.27 18.18 21l-1.64-7.03L22 9.24l-7.19-.61z"/>
</svg>`;

// State
let currentModel = null;
let conversationHistory = [];
let isStreaming = false;
let currentConversationId = null;
let conversations = [];
let sidebarOpen = true;

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

// Conversation sidebar elements
const sidebar = document.getElementById('sidebar');
const sidebarBackdrop = document.getElementById('sidebar-backdrop');
const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebarClose = document.getElementById('sidebar-close');
const conversationList = document.getElementById('conversation-list');
const newConversationBtn = document.getElementById('new-conversation-btn');

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
  initSidebar();
  await checkConnection();
  await loadModels();
  await loadConversations();
  setupEventListeners();
  autoResizeTextarea();
});

// Initialize sidebar state
function initSidebar() {
  const savedState = localStorage.getItem('sidebarOpen');

  // Default to closed, use saved state if available
  if (savedState !== null) {
    sidebarOpen = savedState === 'true';
  } else {
    sidebarOpen = false;
  }

  updateSidebarState();
}

// Toggle sidebar
function toggleSidebar() {
  sidebarOpen = !sidebarOpen;
  updateSidebarState();
  localStorage.setItem('sidebarOpen', sidebarOpen);
}

// Update sidebar visual state
function updateSidebarState() {
  const isMobile = window.innerWidth < 768;
  sidebar.classList.toggle('collapsed', !sidebarOpen);
  sidebarBackdrop.classList.toggle('visible', sidebarOpen && isMobile);
}

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
  document.getElementById('status').title = text;
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
    const availableNames = availableModels.map(m => m.name);

    // Clear and populate select
    modelSelect.innerHTML = '';

    // Find configured models that aren't pulled yet
    const notPulledModels = configuredModels.filter(configName =>
      !availableNames.some(available => available === configName || available.startsWith(configName + ':'))
    );

    // If no models available and we have configured models not pulled, show them
    if (availableModels.length === 0 && notPulledModels.length > 0) {
      const option = document.createElement('option');
      option.value = '';
      option.textContent = 'No models pulled yet - select one to download';
      modelSelect.appendChild(option);
    } else if (availableModels.length === 0) {
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

    // Add separator and not-pulled configured models
    if (notPulledModels.length > 0 && availableModels.length > 0) {
      const separator = document.createElement('option');
      separator.disabled = true;
      separator.textContent = '── Not yet downloaded ──';
      modelSelect.appendChild(separator);
    }

    notPulledModels.forEach(modelName => {
      const option = document.createElement('option');
      option.value = `pull:${modelName}`;
      option.textContent = `⬇ ${modelName} (click to download)`;
      option.className = 'not-pulled';
      modelSelect.appendChild(option);
    });

    // Select default model if available
    if (defaultModel && availableModels.some(m => m.name === defaultModel || m.name.startsWith(defaultModel))) {
      const match = availableModels.find(m => m.name === defaultModel || m.name.startsWith(defaultModel));
      if (match) modelSelect.value = match.name;
    }

    currentModel = modelSelect.value;
    sendBtn.disabled = !currentModel || currentModel.startsWith('pull:');

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

// ==========================================
// Conversation Persistence Functions
// ==========================================

// Load all conversations from server
async function loadConversations() {
  try {
    const response = await fetch('/api/conversations');
    if (response.ok) {
      conversations = await response.json();
      renderConversationList();
      // Load most recent conversation if exists
      if (conversations.length > 0 && !currentConversationId) {
        await loadConversation(conversations[0].id);
      }
    }
  } catch (e) {
    console.log('Conversation persistence not available (running without server.py)');
  }
}

// Render conversation list in sidebar
function renderConversationList() {
  if (!conversationList) return;

  if (conversations.length === 0) {
    conversationList.innerHTML = '<div class="no-conversations">No conversations yet</div>';
    return;
  }

  conversationList.innerHTML = conversations.map(conv => `
    <div class="conversation-item ${conv.id === currentConversationId ? 'active' : ''}"
         data-id="${conv.id}">
      <div class="conversation-title">${escapeHtml(conv.title)}</div>
      <div class="conversation-meta">
        <span>${conv.message_count || 0} messages</span>
        <button class="conversation-delete" data-id="${conv.id}" title="Delete">x</button>
      </div>
    </div>
  `).join('');

  // Add click handlers
  conversationList.querySelectorAll('.conversation-item').forEach(item => {
    item.addEventListener('click', async (e) => {
      if (e.target.classList.contains('conversation-delete')) {
        e.stopPropagation();
        await deleteConversation(e.target.dataset.id);
        return;
      }
      await loadConversation(item.dataset.id);
    });
  });
}

// Load a specific conversation
async function loadConversation(id) {
  try {
    const response = await fetch(`/api/conversations/${id}`);
    if (!response.ok) throw new Error('Failed to load conversation');

    const conv = await response.json();
    currentConversationId = conv.id;
    conversationHistory = conv.messages.map(m => ({ role: m.role, content: m.content }));

    // Update UI
    renderConversationList();
    renderMessages();

    // Update model if conversation has one
    if (conv.model && modelSelect) {
      const option = Array.from(modelSelect.options).find(o => o.value === conv.model);
      if (option) {
        modelSelect.value = conv.model;
        currentModel = conv.model;
      }
    }
  } catch (e) {
    console.error('Error loading conversation:', e);
  }
}

// Create a new conversation
async function createNewConversation() {
  try {
    const response = await fetch('/api/conversations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: currentModel })
    });

    if (!response.ok) throw new Error('Failed to create conversation');

    const conv = await response.json();
    currentConversationId = conv.id;
    conversationHistory = [];
    clearMessages();
    await loadConversations();
  } catch (e) {
    console.error('Error creating conversation:', e);
    // Fallback: just clear local state
    currentConversationId = null;
    conversationHistory = [];
    clearMessages();
  }
}

// Save a message to the current conversation
async function saveMessage(role, content) {
  if (!currentConversationId) return;

  try {
    await fetch(`/api/conversations/${currentConversationId}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ role, content })
    });
  } catch (e) {
    console.error('Error saving message:', e);
  }
}

// Delete a conversation
async function deleteConversation(id) {
  try {
    await fetch(`/api/conversations/${id}`, { method: 'DELETE' });

    if (currentConversationId === id) {
      currentConversationId = null;
      conversationHistory = [];
      clearMessages();
    }

    await loadConversations();

    // Load another conversation if available
    if (!currentConversationId && conversations.length > 0) {
      await loadConversation(conversations[0].id);
    }
  } catch (e) {
    console.error('Error deleting conversation:', e);
  }
}

// Clear messages from UI
function clearMessages() {
  messagesDiv.innerHTML = `
    <div class="welcome-message">
      <h2>Welcome to Ollama Chat</h2>
      <p>Select a model above and start chatting. Your conversation runs on a remote GPU.</p>
      <div class="endpoints-info">
        <div class="endpoint">
          <span class="endpoint-label">API Endpoint:</span>
          <code>http://localhost:11434</code>
        </div>
        <div class="endpoint">
          <span class="endpoint-label">OpenAI-compatible:</span>
          <code>/v1/chat/completions</code>
        </div>
      </div>
    </div>
  `;
}

// Render all messages from current conversation
function renderMessages() {
  messagesDiv.innerHTML = '';
  conversationHistory.forEach(msg => {
    addMessage(msg.role, msg.content);
  });
}

// ==========================================
// Event Listeners
// ==========================================

// Setup event listeners
function setupEventListeners() {
  // Sidebar toggle (open)
  if (sidebarToggle) {
    sidebarToggle.addEventListener('click', toggleSidebar);
  }

  // Sidebar close button
  if (sidebarClose) {
    sidebarClose.addEventListener('click', toggleSidebar);
  }

  // Sidebar backdrop click (close on mobile)
  if (sidebarBackdrop) {
    sidebarBackdrop.addEventListener('click', () => {
      sidebarOpen = false;
      updateSidebarState();
      localStorage.setItem('sidebarOpen', sidebarOpen);
    });
  }

  // Handle window resize
  window.addEventListener('resize', () => {
    updateSidebarState();
  });

  // New conversation button
  if (newConversationBtn) {
    newConversationBtn.addEventListener('click', async () => {
      await createNewConversation();
    });
  }

  // Model selection
  modelSelect.addEventListener('change', async (e) => {
    const value = e.target.value;

    // If user selected a not-pulled model, trigger download
    if (value.startsWith('pull:')) {
      const modelName = value.substring(5); // Remove 'pull:' prefix
      modelNameInput.value = modelName;
      showPullModal();
      // Reset selection to previous or first available
      modelSelect.value = currentModel || '';
      return;
    }

    currentModel = value;
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

  // Create conversation if none exists
  if (!currentConversationId) {
    await createNewConversation();
  }

  // Clear welcome message if present
  const welcome = messagesDiv.querySelector('.welcome-message');
  if (welcome) welcome.remove();

  // Add user message
  addMessage('user', message);
  userInput.value = '';
  userInput.style.height = 'auto';

  // Add to history and save
  conversationHistory.push({ role: 'user', content: message });
  await saveMessage('user', message);

  // Show loading state
  isStreaming = true;
  sendBtn.disabled = true;
  sendBtn.querySelector('.send-icon').classList.add('hidden');
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

    // Add to history, save, and trim if needed
    conversationHistory.push({ role: 'assistant', content: fullResponse });
    await saveMessage('assistant', fullResponse);
    await loadConversations();  // Refresh list to show updated title
    trimHistory();

  } catch (e) {
    console.error('Error sending message:', e);
    contentDiv.innerHTML = `<span class="error">Error: ${escapeHtml(e.message)}</span>`;
  } finally {
    isStreaming = false;
    sendBtn.disabled = false;
    sendBtn.querySelector('.send-icon').classList.remove('hidden');
    sendBtn.querySelector('.btn-loading').classList.add('hidden');
  }
}

// Add message to chat
function addMessage(role, content) {
  const div = document.createElement('div');
  div.className = `message ${role}`;
  const avatar = role === 'user' ? USER_AVATAR : AI_AVATAR;
  div.innerHTML = `
    <div class="message-avatar">${avatar}</div>
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
