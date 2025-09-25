from dotenv import load_dotenv
import streamlit as st
import asyncio
import os
from typing import List

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)

from rag_agent import agent, RAGDeps
from utils import get_chroma_client

load_dotenv()

# Message management configuration
MAX_MESSAGES = 10  # Maximum number of messages to keep in history
MAX_TOKENS_ESTIMATE = 4000  # Rough estimate of token limit for context

async def get_agent_deps():
    return RAGDeps(
        chroma_client=get_chroma_client("./chroma_db"),
        collection_name="docs",
        embedding_model="all-MiniLM-L6-v2"
    )


def estimate_message_tokens(messages: List) -> int:
    """Rough estimation of tokens in message history."""
    total_chars = 0
    for msg in messages:
        if isinstance(msg, (ModelRequest, ModelResponse)):
            for part in msg.parts:
                if hasattr(part, 'content'):
                    total_chars += len(str(part.content))
    # Rough estimate: 4 characters per token
    return total_chars // 4


def limit_message_history(messages: List, max_messages: int = None) -> List:
    """
    Limit message history to the last N messages while preserving pairs.
    Simple truncation - validation happens separately.
    """
    if max_messages is None:
        max_messages = st.session_state.get('custom_max_messages', MAX_MESSAGES)
        
    if len(messages) <= max_messages:
        return messages
    
    # Simple truncation from the end
    return messages[-max_messages:]


def clear_chat_history():
    """Clear the chat history from session state, but preserve complete tool sequences."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.rerun()
        return
    
    # Don't preserve any tool messages to avoid sequence issues
    # Tool calls must be followed by tool returns in proper sequence
    # Clearing all messages ensures we start fresh without orphaned tool messages
    st.session_state.messages = []
    st.rerun()


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # user-prompt
    if part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)
    # tool-call
    elif part.part_kind == 'tool-call':
        with st.chat_message("assistant"):
            with st.expander(f"🔧 Tool Call: {part.tool_name}", expanded=False):
                st.code(str(part.args), language="json")
    # tool-return  
    elif part.part_kind == 'tool-return':
        with st.chat_message("assistant"):
            with st.expander(f"📋 Tool Result: {part.tool_name}", expanded=False):
                st.text(str(part.content))             

def validate_message_history(messages: List) -> List:
    """
    Validate and clean message history to ensure proper tool call sequences.
    Remove any orphaned tool messages that could cause API errors.
    """
    if not messages:
        return messages
    
    validated_messages = []
    tool_calls_stack = []  # Track pending tool calls
    
    for msg in messages:
        if isinstance(msg, (ModelRequest, ModelResponse)):
            msg_tool_calls = []
            msg_tool_returns = []
            other_parts = []
            
            # Categorize parts in this message
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    msg_tool_calls.append(part)
                elif isinstance(part, ToolReturnPart):
                    msg_tool_returns.append(part)
                else:
                    other_parts.append(part)
            
            # Handle tool calls - add to stack
            if msg_tool_calls:
                tool_calls_stack.extend(msg_tool_calls)
                # Keep the message with tool calls
                validated_messages.append(msg)
            
            # Handle tool returns
            elif msg_tool_returns:
                # Only keep tool returns if we have pending tool calls
                if tool_calls_stack:
                    # Remove corresponding tool calls from stack
                    for _ in msg_tool_returns:
                        if tool_calls_stack:
                            tool_calls_stack.pop(0)
                    validated_messages.append(msg)
                # Skip orphaned tool returns
            
            # Handle regular messages (no tool parts)
            elif other_parts:
                validated_messages.append(msg)
    
    return validated_messages

async def run_agent_with_streaming(user_input):
    # Limit message history before passing to agent
    limited_messages = limit_message_history(st.session_state.messages)
    
    # Validate message history to prevent tool sequence errors
    validated_messages = validate_message_history(limited_messages)
    
    # Debug logging
    # print(f"Original messages: {len(st.session_state.messages)}")
    # print(f"Limited messages: {len(limited_messages)}")
    # print(f"Validated messages: {len(validated_messages)}")
    
    async with agent.run_stream(
        user_input, deps=st.session_state.agent_deps, message_history=validated_messages
    ) as result:
        async for message in result.stream_text(delta=True):  
            yield message

    # Add the new messages to the chat history (including tool calls and responses)
    st.session_state.messages.extend(result.new_messages())
    
    # Apply message limiting after adding new messages
    st.session_state.messages = limit_message_history(st.session_state.messages)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~ Main Function with UI Creation ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def main():
    st.title("DocuBot AI")

    # Sidebar for conversation management
    with st.sidebar:
        st.header("💬 Conversation Management")
        
        # Display conversation metrics
        message_count = len(st.session_state.get('messages', []))
        estimated_tokens = estimate_message_tokens(st.session_state.get('messages', []))
        current_limit = st.session_state.get('custom_max_messages', MAX_MESSAGES)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", f"{message_count}/{current_limit}")
        with col2:
            st.metric("Est. Tokens", estimated_tokens)
        
        # Progress bar for message limit
        progress = min(message_count / current_limit, 1.0)
        st.progress(progress, text=f"Message History Usage")
        
        # Warning if approaching limits
        if message_count > current_limit * 0.8:
            st.warning("⚠️ Approaching message limit. Older messages will be automatically removed.")
        
        if estimated_tokens > MAX_TOKENS_ESTIMATE * 0.8:
            st.warning("⚠️ Approaching token limit. Consider clearing history.")
        
        # Clear history button
        if st.button("🗑️ Clear Chat History", help="Clear all conversation history"):
            clear_chat_history()
        
        st.divider()
        
        # Settings
        st.header("⚙️ Settings")
        
        # Customizable message limit
        if "custom_max_messages" not in st.session_state:
            st.session_state.custom_max_messages = MAX_MESSAGES
            
        new_limit = st.slider(
            "Max Messages to Keep",
            min_value=3,
            max_value=20,
            value=st.session_state.custom_max_messages,
            step=5,
            help="Maximum number of messages to keep in conversation history"
        )
        
        if new_limit != st.session_state.custom_max_messages:
            st.session_state.custom_max_messages = new_limit
            # Apply new limit immediately
            st.session_state.messages = limit_message_history(
                st.session_state.messages, new_limit
            )
            st.rerun()
        
        st.write("**Features:**")
        st.write(f"• Automatic limiting: ✅ Enabled")
        st.write(f"• Preserves conversation pairs: ✅ Yes")
        st.write(f"• Token estimation: ✅ Enabled")
        
        # Information expander
        with st.expander("ℹ️ About Message Management"):
            st.write("""
            **Why limit messages?**
            - Prevents memory issues in long conversations
            - Keeps response times fast
            - Avoids token limits with AI models
            
            **How it works:**
            - Automatically keeps the most recent messages
            - Preserves question-answer pairs
            - Shows warnings when approaching limits
            - Estimates token usage for AI model
            
            **Tips:**
            - Clear history for fresh start
            - Adjust limit based on your needs
            - Watch the progress bar
            """)

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_deps" not in st.session_state:
        st.session_state.agent_deps = await get_agent_deps()  

    # Display all messages from the conversation so far
    # Each message is either a ModelRequest or ModelResponse.
    # We iterate over their parts to decide how to display them.
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input("What do you want to know?")

    if user_input:
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Create a placeholder for the streaming text
            message_placeholder = st.empty()
            full_response = ""
            
            # Properly consume the async generator with async for
            generator = run_agent_with_streaming(user_input)
            async for message in generator:
                full_response += message
                message_placeholder.markdown(full_response + "▌")
            
            # Final response without the cursor
            message_placeholder.markdown(full_response)


if __name__ == "__main__":
    asyncio.run(main())