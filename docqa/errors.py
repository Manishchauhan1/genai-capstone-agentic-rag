class DocQAError(Exception):
    """A failure whose message is safe to show to the user.

    Never put secrets (API keys) or stack traces in the message.
    """
