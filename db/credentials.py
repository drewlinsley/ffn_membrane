def postgresql_credentials():
    """Credentials for your psql DB."""
    return {
        'username': 'connectomics',
        'password': 'connectomics',
        'database': 'connectomics'
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

