from llmcore.config.models import validate_semantiscan_config

# Minimal config should validate
config_dict = {
    'enabled': True,
    'database': {'type': 'chromadb', 'path': '~/.llmcore/chroma_db', 'collection_name': 'test'},
    # ... (see test file for complete minimal config)
}

validated = validate_semantiscan_config(config_dict)
print(f"Validation successful: {validated.database.type}")
