"""Application configuration loader — messy, needs refactoring."""
import os
import json

def load_config():
    config = {}

    # Database
    config['db_host'] = os.environ.get('DB_HOST', 'localhost')
    config['db_port'] = int(os.environ.get('DB_PORT', '5432'))
    config['db_name'] = os.environ.get('DB_NAME', 'myapp')
    config['db_user'] = os.environ.get('DB_USER', 'postgres')
    config['db_password'] = os.environ.get('DB_PASSWORD', '')
    config['db_pool_size'] = int(os.environ.get('DB_POOL_SIZE', '10'))
    config['db_pool_timeout'] = int(os.environ.get('DB_POOL_TIMEOUT', '30'))

    # Redis
    config['redis_host'] = os.environ.get('REDIS_HOST', 'localhost')
    config['redis_port'] = int(os.environ.get('REDIS_PORT', '6379'))
    config['redis_password'] = os.environ.get('REDIS_PASSWORD', '')
    config['redis_db'] = int(os.environ.get('REDIS_DB', '0'))

    # App
    config['app_name'] = os.environ.get('APP_NAME', 'MyApp')
    config['app_env'] = os.environ.get('APP_ENV', 'development')
    config['app_debug'] = os.environ.get('APP_DEBUG', 'true').lower() == 'true'
    config['app_port'] = int(os.environ.get('APP_PORT', '8000'))
    config['app_host'] = os.environ.get('APP_HOST', '0.0.0.0')
    config['app_workers'] = int(os.environ.get('APP_WORKERS', '4'))
    config['app_secret'] = os.environ.get('APP_SECRET', 'changeme')

    # Logging
    config['log_level'] = os.environ.get('LOG_LEVEL', 'INFO')
    config['log_format'] = os.environ.get('LOG_FORMAT', 'json')
    config['log_file'] = os.environ.get('LOG_FILE', '')

    # Email
    config['smtp_host'] = os.environ.get('SMTP_HOST', 'localhost')
    config['smtp_port'] = int(os.environ.get('SMTP_PORT', '587'))
    config['smtp_user'] = os.environ.get('SMTP_USER', '')
    config['smtp_password'] = os.environ.get('SMTP_PASSWORD', '')
    config['smtp_tls'] = os.environ.get('SMTP_TLS', 'true').lower() == 'true'
    config['email_from'] = os.environ.get('EMAIL_FROM', 'noreply@example.com')

    # Feature flags
    config['feature_new_ui'] = os.environ.get('FEATURE_NEW_UI', 'false').lower() == 'true'
    config['feature_api_v2'] = os.environ.get('FEATURE_API_V2', 'false').lower() == 'true'
    config['feature_websockets'] = os.environ.get('FEATURE_WEBSOCKETS', 'false').lower() == 'true'

    # Validate
    if config['app_env'] not in ('development', 'staging', 'production'):
        raise ValueError(f"Invalid APP_ENV: {config['app_env']}")
    if config['app_env'] == 'production' and config['app_secret'] == 'changeme':
        raise ValueError("APP_SECRET must be set in production")
    if config['app_env'] == 'production' and config['app_debug']:
        raise ValueError("APP_DEBUG must be false in production")

    # Load overrides from file
    config_file = os.environ.get('CONFIG_FILE', '')
    if config_file and os.path.exists(config_file):
        with open(config_file) as f:
            overrides = json.load(f)
            config.update(overrides)

    return config


def get_db_url(config):
    return f"postgresql://{config['db_user']}:{config['db_password']}@{config['db_host']}:{config['db_port']}/{config['db_name']}"


def get_redis_url(config):
    if config['redis_password']:
        return f"redis://:{config['redis_password']}@{config['redis_host']}:{config['redis_port']}/{config['redis_db']}"
    return f"redis://{config['redis_host']}:{config['redis_port']}/{config['redis_db']}"
