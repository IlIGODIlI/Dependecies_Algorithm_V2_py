# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 2.0.x   | Yes       |
| < 2.0   | No        |

## Reporting a Vulnerability

If you discover a security vulnerability in Vela, please report it responsibly.

**Do NOT open a public issue.**

Instead, please email the maintainer directly with:

1. A description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fix (if any)

We will acknowledge receipt within 48 hours and aim to release a fix within 7 days for critical issues.

## Security Considerations

### Ollama Connection

Vela communicates with a locally running Ollama instance via HTTP on `localhost:11434`. This connection is unencrypted and trusts the local machine. Do not expose Ollama's port to untrusted networks.

### Model Persistence

Model data is stored as plain JSON files in the `models/` directory. These files may contain sensitive training data. Ensure appropriate filesystem permissions in production environments.

### API CORS

The default FastAPI configuration allows all origins (`allow_origins=["*"]`). **Tighten this in production** to only allow trusted domains:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### File Training Endpoint

The `POST /train/file` endpoint reads files from the server filesystem. In production, validate and restrict allowed file paths to prevent path traversal attacks.
