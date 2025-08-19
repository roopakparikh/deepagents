
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

