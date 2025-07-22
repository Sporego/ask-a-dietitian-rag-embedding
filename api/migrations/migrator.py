import datetime
import glob
import hashlib
import logging
import os
import os.path
from collections import OrderedDict
from importlib import import_module

import psycopg

log = logging.getLogger(__name__)

schema = '''
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE migrations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created TIMESTAMP DEFAULT NOW(),
    name TEXT NOT NULL,
    size INTEGER NOT NULL,
    ctime TIMESTAMP NOT NULL,
    mtime TIMESTAMP NOT NULL,
    sha256 TEXT NOT NULL
);
'''



def get_file_migrations():
    here = os.path.dirname(os.path.abspath(__file__))
    files = glob.glob(os.path.join(here, '*.sql')) + glob.glob(os.path.join(here, '*.py'))
    migrations = OrderedDict()
    for path in sorted(files):
        name = os.path.split(path)[1]
        if not name[0].isdigit():
            continue
        st = os.stat(path)
        migrations[name] = {
            'name': name,
            'path': path,
            'size': st.st_size,
            'ctime': datetime.datetime.utcfromtimestamp(st.st_ctime),
            'mtime': datetime.datetime.utcfromtimestamp(st.st_mtime),
            'sha256': hashlib.sha256(open(path, encoding='utf8').read().encode('utf8')).hexdigest(),
        }
    return migrations


def get_db_migrations(conn: psycopg.Connection):
    try:
        rows = conn.execute('select * from migrations').fetchall()
    except:
        rows = []
        conn.rollback()
        log.info('Creating migration table')
        conn.execute(schema)
        conn.commit()

    return rows


def run_migrations(conn: psycopg.Connection):
    migrations = get_file_migrations()
    applied = get_db_migrations(conn)

    # check and remove any already applied migrations
    for migration in applied:
        found = migrations.pop(migration['name'], None)
        assert found, ('Deleted migration?', migration)
        assert migration['sha256'] == found['sha256'], ('Checksum mismatch', migration, found)

    if not migrations:
        log.info('Nothing to do!')
        return

    # apply any leftover migrations
    for migration in migrations.values():
        if migration['name'].endswith('.sql'):
            log.info('Applying %s', migration['name'])
            q = open(migration['path'], encoding='utf8').read()
            conn.execute(q)
        else:
            log.info('Applying %s', migration['name'])
            mod = import_module('.' + migration['name'].split('.')[0], package='migrations')
            mod.migrate(conn)

        q = 'insert into migrations (name, size, ctime, mtime, sha256) values (%s, %s, %s, %s, %s)'
        values = tuple(migration[k] for k in ('name', 'size', 'ctime', 'mtime', 'sha256'))
        conn.execute(q, values)

    # all good
    conn.commit()
    log.info('Commit!')
