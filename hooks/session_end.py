#!/usr/bin/env python3
"""Stop hook — nudge Claude to auto-extract memories and log the session."""

import json
import sys

message = {
    "systemMessage": (
        "Session ending. Please:\n"
        "1. Call extract_memories with a comprehensive summary of this session covering: "
        "key decisions made, user preferences discovered, architectural choices, "
        "debugging insights, project context, and any learnings worth remembering "
        "across sessions. Be thorough — this is how you'll remember this conversation.\n"
        "2. Call log_session with a brief summary of what was accomplished."
    )
}

print(json.dumps(message))
