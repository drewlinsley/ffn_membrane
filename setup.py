import os
# from setuptools import setup
from db import credentials
from utils import logger
from config import Config


"""e.g. python setup.py install"""
try:
    from pip.req import parse_requirements
    install_reqs = parse_requirements('requirements.txt', session='hack')
    # reqs is a list of requirement
    # e.g. ['django==1.5.1', 'mezzanine==1.4.6']
    reqs = [str(ir.req) for ir in install_reqs]
except Exception:
    print('Failed to import parse_requirements.')


# try:
#     setup(
#         name="membrane_filling",
#         version="0.0.1",
#         packages=install_reqs,
#     )
# except Exception as e:
#     print(
#         'Failed to install requirements and compile repo. '
#         'Try manually installing. %s' % e)

config = Config()
log = logger.get(os.path.join(config.log_dir, 'setup'))
log.info('Installed required packages and created paths.')

params = credentials.postgresql_connection()
sys_password = credentials.machine_credentials()['password']
os.popen(
    'sudo -u postgres createuser -sdlP %s' % params['user'], 'w').write(
    sys_password)
os.popen(
    'sudo -u postgres createdb %s -O %s' % (
        params['database'],
        params['user']), 'w').write(sys_password)
log.info('Created DB.')
