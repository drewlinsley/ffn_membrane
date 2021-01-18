def postgresql_credentials():
    """Credentials for your psql DB."""
    return {
        'username': 'connectomics',
        'password': 'connectomics',
        'database': 'connectomics'
    }


def machine_credentials():
    """Credentials for your machine."""
    return {
        'username': 'drew',
        'password': 'serrelab',
        'ssh_address': '10.4.128.131'  # 'serrep7.services.brown.edu'
    }


def postgresql_connection(port=''):
    """Package DB credentials into a dictionary."""
    unpw = postgresql_credentials()
    params = {
        'database': unpw['database'],
        'user': unpw['username'],
        'password': unpw['password'],
        'host': 'localhost',
        'port': port,
    }
    return params
