// vLLM Chat - GPU CLI Web UI
// Lightweight chat interface for vLLM OpenAI-compatible API with conversation persistence

const VLLM_API = 'http://localhost:8000';
const MODELS_CONFIG = './models.json';
const MAX_HISTORY_LENGTH = 50;

// State
let currentModel = null;
let conversationHistory = [];
let isStreaming = false;
let currentConversationId = null;
let conversations = [];

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

// Conversation sidebar elements
const conversationList = document.getElementById('conversation-list');
const newConversationBtn = document.getElementById('new-conversation-btn');

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
  await loadModelConfig();
  await checkConnection();
  await loadConversations();
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
      <h2>Welcome to vLLM Chat</h2>
      <p>High-performance LLM inference running on a remote GPU.</p>
      <div class="endpoints-info">
        <div class="endpoint">
          <span class="endpoint-label">API Endpoint:</span>
          <code>http://localhost:8000</code>
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
  // New conversation button
  if (newConversationBtn) {
    newConversationBtn.addEventListener('click', async () => {
      await createNewConversation();
    });
  }

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
