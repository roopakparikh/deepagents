
[![Create Release](https://github.com/roopakparikh/deepagents/actions/workflows/main.yml/badge.svg)](https://github.com/roopakparikh/deepagents/actions/workflows/main.yml)

This code is forked from the  DeepAgents CLI.

# 🧠🤖Deep Agents
The deepagents has a CLI that allows you to interact with the deepagents in an interactive way. The CLI works off of MCP Server that can be configured via a config file, see the [CLI Usage](#cli-usage) section for more details.

The CLI is meant to be whitelabeled for your use case. You can rename the binary to whatever you want, and it will adapt to the name it was invoked as.

# Use
Download the latest binary (a pyinstaller created binary) from the [releases](https://github.com/roopakparikh/deepagents/releases) page.

```bash
wget https://github.com/roopakparikh/deepagents/releases/download/v0.1.0/deepagents-cli
chmod +x deepagents-cli
./deepagents-cli

ln -s /path/to/deepagents-cli /usr/local/bin/youragentname
youragentname
```

**Key CLI Features:**
- 🤖 Interactive chat with deep agent capabilities
- 📁 File path autocompletion triggered by `@` symbol

**CLI Commands:**
- `help` - Show available commands
- `status` - Display current todos and files
- `clear` - Clear terminal screen
- `exit`/`quit` - Exit the CLI

**Debug Options:**
- `--debug` - Enable comprehensive debug logging for all tools and agent execution
- `--debug-mcp` - Enable debug logging for MCP tools only

### Debugging Features

The CLI and tools support comprehensive debug logging to help diagnose issues:

```bash
# Run with debug mode enabled
./deepagents-cli --debug

# Or set environment variable directly
DEEPAGENTS_DEBUG=1 ./deepagents-cli
```

Debug logging provides detailed information about:
- Tool invocations with inputs and outputs
- Execution timing for each tool
- Agent step-by-step execution
- Error stack traces
- MCP tool calls and responses
- Sub-agent creation and invocation

All debug output is sent to stderr, so you can redirect it if needed:
```bash
./deepagents-cli --debug 2>debug.log
```

### Using Debug Utilities in Custom Tools

If you're developing custom tools, you can use the built-in debug utilities:

```python
from deepagents.debug_utils import debug_tool, log_debug, is_debug_enabled

# Automatically log tool inputs, outputs, timing, and errors
@debug_tool
def my_custom_tool(input_data):
    # Check if debug is enabled before expensive operations
    if is_debug_enabled():
        log_debug(f"Processing custom data: {input_data}")
        
    # Your tool implementation
    result = process_data(input_data)
    
    # Log important events
    if is_debug_enabled():
        log_debug(f"Processed {len(result)} items")
        
    return result
```

The debug utilities will only log when debug mode is enabled via the `--debug` flag or `DEEPAGENTS_DEBUG=1` environment variable.

**File Path Completion:**
Type `@` followed by a path and press TAB to autocomplete:
- `@/home/user/doc<TAB>` - Complete absolute paths
- `@src/deep<TAB>` - Complete relative paths
- `@../parent<TAB>` - Complete parent directories

### CLI name and per-name config

The CLI adapts to the name it was invoked as (useful for creating multiple personas/configs):

- **Rename via symlink or copy**
  ```bash
  # Build or install deepagents-cli, then create a symlink with your desired name
  ln -s /path/to/dist/deepagents-cli /usr/local/bin/hapgpt

  # Now help/usage will show `hapgpt`
  hapgpt --help
  ```
  Note: shell aliases do not change the program name seen by the CLI.

- **Default config per program name**
  If `--config` is not provided, the CLI will look for a config at:
  ```
  $HOME/.<program-name>/config.json
  ```
  For example, with the symlink above:
  ```bash
  mkdir -p ~/.hapgpt
  cp example-config.json ~/.hapgpt/config.json
  hapgpt
  ```
  To use a different file explicitly, pass `--config /path/to/config.json`.

