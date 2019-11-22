#!/usr/bin/env python
import os
import json
import sshtunnel
import argparse
import psycopg2
import psycopg2.extras
import psycopg2.extensions
import credentials
import numpy as np
from tqdm import tqdm
from config import Config
sshtunnel.DAEMON = True  # Prevent hanging process due to forward thread
main_config = Config()


class db(object):
    def __init__(self, config):
        """Init global variables."""
        self.status_message = False
        self.db_schema_file = os.path.join('db', 'db_schema.txt')
        # Pass config -> this class
        for k, v in config.items():
            setattr(self, k, v)

    def __enter__(self):
        """Enter method."""
        try:
            if main_config.db_ssh_forward:
                forward = sshtunnel.SSHTunnelForwarder(
                    credentials.machine_credentials()['ssh_address'],
                    ssh_username=credentials.machine_credentials()['username'],
                    ssh_password=credentials.machine_credentials()['password'],
                    remote_bind_address=('127.0.0.1', 5432))
                forward.start()
                self.forward = forward
                self.pgsql_port = forward.local_bind_port
            else:
                self.forward = None
                self.pgsql_port = ''
            pgsql_string = credentials.postgresql_connection(
                str(self.pgsql_port))
            self.pgsql_string = pgsql_string
            self.conn = psycopg2.connect(**pgsql_string)
            self.conn.set_isolation_level(
                psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            self.cur = self.conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor)
        except Exception as e:
            self.close_db()
            if main_config.db_ssh_forward:
                self.forward.close()
            print(e)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit method."""
        if exc_type is not None:
            print exc_type, exc_value, traceback
            self.close_db(commit=False)
        else:
            self.close_db()
        if main_config.db_ssh_forward:
            self.forward.close()
        return self

    def close_db(self, commit=True):
        """Commit changes and exit the DB."""
        self.conn.commit()
        self.cur.close()
        self.conn.close()

    def recreate_db(self):
        """Initialize the DB from the schema file."""
        db_schema = open(self.db_schema_file).read().splitlines()
        for s in db_schema:
            t = s.strip()
            if len(t):
                self.cur.execute(t)

    def return_status(
            self,
            label,
            throw_error=False):
        """
        General error handling and status of operations.
        ::
        label: a string of the SQL operation (e.g. 'INSERT').
        throw_error: if you'd like to terminate execution if an error.
        """
        if label in self.cur.statusmessage:
            print 'Successful %s.' % label
        else:
            if throw_error:
                raise RuntimeError('%s' % self.cur.statusmessag)
            else:
                'Encountered error during %s: %s.' % (
                    label, self.cur.statusmessage
                )

    def reset(self):
        """Reset in coordinate info."""
        self.cur.execute(
            """
            UPDATE
            coordinates SET
            is_processing=False, processed=False, run_number=NULL, chain_id=NULL
            """
        )
        if self.status_message:
            self.return_status('RESET')

    def reset_priority(self):
        """Remove all entries from the priority table."""
        self.cur.execute(
            """
            DELETE FROM priority
            """
        )
        if self.status_message:
            self.return_status('DELETE')

    def reset_config(self):
        """Remove all entries from the config."""
        self.cur.execute(
            """
            DELETE from config
            """
        )
        self.cur.execute(
            """
            INSERT INTO config
            (global_max_id, number_of_segments, max_chain_id)
            VALUES (1, 0, 0)
            """
        )
        if self.status_message:
            self.return_status('RESET')

    def populate_db_with_all_coords_fast(self, values):
        """
        Add a combination of parameter_dict to the db.
        ::
        """
        psycopg2.extras.execute_values(
            """
            INSERT INTO coordinates
            (
                x,
                y,
                z,
                is_processing,
                processed,
                run_number,
                chain_id
            )
            VALUES %s
            """,
            values)
        if self.status_message:
            self.return_status('INSERT')

    def populate_db_with_all_coords(self, namedict, experiment_link=False):
        """
        Add a combination of parameter_dict to the db.
        ::
        experiment_name: name of experiment to add
        experiment_link: linking a child (e.g. clickme) -> parent (ILSVRC12)
        """
        self.cur.executemany(
            """
            INSERT INTO coordinates
            (
                x,
                y,
                z,
                is_processing,
                processed,
                run_number,
                chain_id
            )
            VALUES
            (
                %(x)s,
                %(y)s,
                %(z)s,
                %(is_processing)s,
                %(processed)s,
                %(run_number)s,
                %(chain_id)s
            )
            """,
            namedict)
        if self.status_message:
            self.return_status('INSERT')

    def insert_segments(self, namedict):
        """Insert segment info into segment table."""
        self.cur.executemany(
            """
            INSERT INTO segments
            (
                x,
                y,
                z,
                segment_id,
                size,
                chain_id
            )
            VALUES
            (
                %(x)s,
                %(y)s,
                %(z)s,
                %(segment_id)s,
                %(size)s,
                %(chain_id)s
            )
            """,
            namedict)
        if self.status_message:
            self.return_status('INSERT')

    def create_config(self, namedict, experiment_link=False):
        """
        Add a combination of parameter_dict to the db.
        ::
        experiment_name: name of experiment to add
        experiment_link: linking a child (e.g. clickme) -> parent (ILSVRC12)
        """
        self.cur.executemany(
            """
            INSERT INTO config
            (
                global_max_id,
                number_of_segments,
                max_chain_id,
                total_coordinates,
            )
            VALUES
            (
                %(global_max_id)s,
                %(number_of_segments)s,
                %(max_chain_id)s,
                %(total_coordinates)s,
            )
            """,
            namedict)
        if self.status_message:
            self.return_status('INSERT')

    def get_config(self, experiment_link=False):
        """
        Return config.
        """
        self.cur.execute(
            """
            SELECT * FROM config
            """)
        if self.status_message:
            self.return_status('SELECT')
        return self.cur.fetchone()

    def update_global_max(self, value, experiment_link=False):
        """
        Update current global max segment id.
        """
        self.cur.execute(
            """
            UPDATE config SET global_max_id=%s
            """ %
            str(value))
        if self.status_message:
            self.return_status('UPDATE')

    def update_number_of_segments(self, value):
        """
        Update current number of segments.
        """
        self.cur.execute(
            """
            UPDATE config SET number_of_segments=%s
            """ %
            str(value))
        if self.status_message:
            self.return_status('UPDATE')

    def update_max_chain_id(self, value, experiment_link=False):
        """
        Update current max chain id.
        """
        self.cur.execute(
            """
            UPDATE config SET max_chain_id=%s
            """ %
            str(value))
        if self.status_message:
            self.return_status('UPDATE')

    def add_segments(self, namedict, experiment_link=False):
        """
        Add a combination of parameter_dict to the db.
        ::
        experiment_name: name of experiment to add
        experiment_link: linking a child (e.g. clickme) -> parent (ILSVRC12)
        """
        self.cur.executemany(
            """
            INSERT INTO segments
            (
                segment_id,
                size,
                start_x,
                start_y,
                start_z,
                x,
                y,
                z,
                chain_id,
            )
            VALUES
            (
                %(segment_id)s,
                %(size)s,
                %(start_x)s,
                %(start_y)s,
                %(start_z)s,
                %(x)s,
                %(y)s,
                %(z)s,
                %(chain_id)s,
            )
            """,
            namedict)
        if self.status_message:
            self.return_status('INSERT')

    def add_priorities(self, namedict, experiment_link=False):
        """
        Add a combination of parameter_dict to the db.
        ::
        experiment_name: name of experiment to add
        experiment_link: linking a child (e.g. clickme) -> parent (ILSVRC12)
        """
        self.cur.executemany(
            """
            INSERT INTO priority
            (
                x,
                y,
                z,
                force,
                quality,
                location,
                prev_chain_idx,
                processed,
                chain_id
            )
            VALUES
            (
                %(x)s,
                %(y)s,
                %(z)s,
                %(force)s,
                %(quality)s,
                %(location)s,
                %(prev_chain_idx)s,
                %(processed)s,
                %(chain_id)s
            )
            ON CONFLICT DO NOTHING""",
            namedict)
        if self.status_message:
            self.return_status('INSERT')

    def get_next_priority(self, experiment_link=False):
        """
        Return next row of priority table.
        """
        self.cur.execute(
            """
            UPDATE priority
            SET processed=TRUE
            WHERE _id=(
                SELECT _id
                FROM priority
                WHERE processed=FALSE
                LIMIT 1)
            RETURNING *
            """)
        if self.status_message:
            self.return_status('SELECT')
        return self.cur.fetchone()

    def pull_chain(self, chain_id):
        """Pull a segmentation chain."""
        self.cur.execute(
            """
            SELECT *
            FROM priority
            WHERE chain_id=%s
            """ % chain_id)
        if self.status_message:
            self.return_status('SELECT')
        if self.cur.description is None:
            return None
        else:
            return self.cur.fetchall()

    def reserve_coordinate(self, x, y, z):
        """Set is_processing=True."""
        self.cur.execute(
            """
            UPDATE coordinates
            SET is_processing=TRUE, start_date='now()'
            WHERE x=%s AND y=%s AND z=%s""" % (x, y, z))
        if self.status_message:
            self.return_status('UPDATE')

    def finish_coordinate(self, x, y, z):
        """Set processed=True."""
        self.cur.execute(
            """
            UPDATE coordinates
            SET processed=TRUE, end_date='now()'
            WHERE x=%s AND y=%s AND z=%s""" % (x, y, z))
        if self.status_message:
            self.return_status('UPDATE')

    def check_coordinate(self, namedict):
        """Test coordinates for segmenting/not."""
        self.cur.executemany(
            """
            SELECT _id
            FROM coordinates
            WHERE (
                (processed=TRUE) OR
                (is_processing=TRUE AND
                DATE_PART('day', start_date - 'now()') = 0))
            AND x=%(x)s AND y=%(y)s AND z=%(z)s
            """,
            namedict)
        if self.status_message:
            self.return_status('SELECT')
        if self.cur.description is None:
            return None
        else:
            return self.cur.fetchall()

    def get_coordinate(self, experiment=None, random=True):
        """After returning coordinate, set processing=True."""
        self.cur.execute(
            """
            UPDATE coordinates
            SET is_processing=TRUE, date='now()'
            WHERE _id=(
                SELECT _id
                FROM coordinates
                WHERE (processed=FALSE AND is_processing=FALSE)
                OR (processed=FALSE AND DATE_PART('day', date - 'now()') > 0)
                ORDER BY random()
                LIMIT 1)
            RETURNING *
            """)
        if self.status_message:
            self.return_status('SELECT')
        return self.cur.fetchone()

    def get_total_coordinates(self):
        """Return the count of coordinates."""
        self.cur.execute(
            """
            SELECT count(*)
            FROM coordinates
            """)
        if self.status_message:
            self.return_status('SELECT')
        return self.cur.fetchone()

    def get_finished_coordinates(self):
        """Return the count of finished coordinates."""
        self.cur.execute(
            """
            SELECT count(*)
            FROM coordinates
            WHERE processed=True
            """)
        if self.status_message:
            self.return_status('SELECT')
        return self.cur.fetchone()


def initialize_database():
    """Initialize and recreate the database."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        db_conn.recreate_db()
        db_conn.return_status('CREATE')


def reset_database():
    """Reset coordinate progress."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        db_conn.reset()
        db_conn.return_status('RESET')


def reset_priority():
    """Reset priority list."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        db_conn.reset_priority()
        db_conn.return_status('RESET')


def reset_config():
    """Reset global config."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        db_conn.reset_config()
        db_conn.return_status('RESET')


def populate_db(coords, fast=True):
    """Add coordinates to DB."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        coord_dict = []
        for coord in tqdm(
                coords,
                total=len(coords),
                desc='Processing coordinates'):
            split_coords = coord.split(os.path.sep)
            x = [x.strip('x').lstrip('0') for x in split_coords if 'x' in x][0]
            y = [y.strip('y').lstrip('0') for y in split_coords if 'y' in y][0]
            z = [z.strip('z').lstrip('0') for z in split_coords if 'z' in z][0]
            if not len(x):
                x = '0'
            if not len(y):
                y = '0'
            if not len(z):
                z = '0'
            if fast:
                coord_dict += [{
                    'x': int(x),
                    'y': int(y),
                    'z': int(z),
                    'is_processing': False,
                    'processed': False,
                    'run_number': None,
                    'chain_id': None}]
            else:
                coord_dict += [
                    int(x),
                    int(y),
                    int(z),
                    False,
                    False,
                    None,
                    None]
        print('Populating DB')
        if fast:
            db_conn.populate_db_with_all_coords(coord_dict)
        else:
            db_conn.populate_db_with_all_coords_fast(coord_dict)
        db_conn.return_status('CREATE')


def add_priorities(priorities):
    """Add priority coordinates to DB."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        priority_dict = []
        for idx, p in tqdm(
                priorities.iterrows(),
                total=len(priorities),
                desc='Processing priorities'):
            priority_dict += [{
                'x': int(p.x),
                'y': int(p.y),
                'z': int(p.z),
                'quality': p.quality,
                'location': p.location,
                'force': p.force,
                'prev_chain_idx': p.prev_chain_idx,
                'processed': False,
                'chain_id': p.chain_id,
            }]
        db_conn.add_priorities(priority_dict)
        db_conn.return_status('CREATE')


def get_global_max():
    """Get global max id from config."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        global_max = db_conn.get_config()
        db_conn.return_status('SELECT')
    assert global_max is not None, 'You may need to reset the config.'
    return global_max['global_max_id']


def update_global_max(value):
    """Get global max id from config."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        db_conn.update_global_max(value)
        db_conn.return_status('UPDATE')


def update_config_segments_chain(value):
    """Add to the segment counter."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        cfg = db_conn.get_config()
        db_conn.update_number_of_segments(value + cfg['number_of_segments'])
        db_conn.return_status('UPDATE')


def update_max_chain_id(value):
    """Get global max id from config."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        db_conn.update_max_chain_id(value)
        db_conn.return_status('UPDATE')


def get_next_priority():
    """Grab next row from priority table."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        priority = db_conn.get_next_priority()
        db_conn.return_status('SELECT')
    return priority


def insert_segments(segment_dicts):
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        db_conn.insert_segments(segment_dicts)
        db_conn.return_status('INSERT')


def get_coordinate():
    """Grab next row from coordinate table."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        coordinate = db_conn.get_coordinate()
        db_conn.return_status('SELECT')
    return coordinate


def reserve_coordinate(x, y, z):
    """Reserve coordinate from coordinate table."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        db_conn.reserve_coordinate(x=x, y=y, z=z)
        db_conn.return_status('UPDATE')


def check_coordinate(coordinate):
    """Return coordinates if they pass the test."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        res = db_conn.check_coordinate(coordinate)
        db_conn.return_status('SELECT')
    return res


def finish_coordinate(x, y, z):
    """Finish off the coordinate from coordinate table."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        db_conn.finish_coordinate(x=x, y=y, z=z)
        db_conn.return_status('UPDATE')


def lookup_chain(chain_id, prev_chain_idx):
    """Pull the chain_id then get cooridnate of the prev_chain_idx."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        chained_coordinates = db_conn.pull_chain(chain_id)
    if chained_coordinates is not None:
        ids = [
            x['prev_chain_idx'] if x['prev_chain_idx'] is not None else 0
            for x in chained_coordinates]
        prev_idx = ids.index(prev_chain_idx)
        r = chained_coordinates[prev_idx]
        return (r['x'], r['y'], r['z'])
    return None


def get_next_coordinate(path_extent, stride):
    """Get next coordinate to process.
    First pull and delete from priority list.
    If nothing there, select a random coordinate from all coords."""
    result = get_next_priority()
    if result is None:
        result = get_coordinate()
        priority = False
    else:
        reserve_coordinate(x=result['x'], y=result['y'], z=result['z'])
        priority = True
    x = result['x']
    y = result['y']
    z = result['z']
    chain_id = result['chain_id']

    # Find the previous coordinate from a priority
    prev_chain_idx = result.get('prev_chain_idx', False)
    if prev_chain_idx is not None and prev_chain_idx > 0:
        prev_chain_idx -= 1
    else:
        prev_chain_idx = None
    chain_id = result.get('chain_id', False)
    if chain_id and prev_chain_idx is not None:
        prev_coordinate = lookup_chain(
            chain_id=chain_id,
            prev_chain_idx=prev_chain_idx)
    else:
        prev_coordinate = None

    # Check that we need to segment this coordinate
    check_rng = np.array(path_extent) - np.array(stride)
    x_range = range(x - check_rng[0], x + check_rng[0])
    y_range = range(y - check_rng[1], y + check_rng[1])
    z_range = range(z - check_rng[2], z + check_rng[2])
    xyzs = []
    if check_rng[0]:
        for xid in x_range:
            xyzs += [{
                'x': xid,
                'y': y,
                'z': z}]
    if check_rng[1]:
        for yid in y_range:
            xyzs += [{
                'x': x,
                'y': yid,
                'z': z}]
    if check_rng[2]:
        for zid in z_range:
            xyzs += [{
                'x': x,
                'y': y,
                'z': zid}]
    if np.any(check_rng > 0):
        xyz_checks = check_coordinate(xyzs)
    else:
        xyz_checks = None
    force = result.get('force', False)
    if force:
        xyz_checks = None
    if prev_chain_idx is None:
        prev_chain_idx = 0
    if xyz_checks is None:
        if chain_id is None:
            chain_id = get_max_chain_id() + 1
            update_max_chain_id(chain_id)
        return (x, y, z, chain_id, prev_chain_idx, priority, (prev_coordinate))


def adjust_max_id(segmentation):
    """Look into the global config to adjust ids."""
    max_id = 0
    try:
        max_id = get_global_max()
    except Exception as e:
        print('Failed to access db: %s' % e)
    segmentation_mask = (segmentation > 0).astype(segmentation.dtype)
    segmentation += (segmentation_mask * max_id)
    try:
        update_global_max(segmentation.max())
    except Exception as e:
        print('Failed to update db global max: %s' % e)
    return segmentation


def get_max_chain_id():
    """Pull max_chain_id from config."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        global_max = db_conn.get_config()
        db_conn.return_status('SELECT')
    assert global_max is not None, 'You may need to reset the config.'
    return global_max['max_chain_id']


def get_progress(extent=[2, 3, 3]):
    """Get percent finished of the whole connectomics volume."""
    config = credentials.postgresql_connection()
    with db(config) as db_conn:
        total_segments = db_conn.get_total_coordinates()['count']
        finished_segments = db_conn.get_finished_coordinates()['count']
        finished_segments *= np.prod(extent)
        prop_finished = float(finished_segments) / float(total_segments)
        print('Segmentation is {}% complete.'.format(prop_finished * 100))
    return prop_finished


def get_performance(experiment_name, force_fwd=False):
    """Get performance for an experiment."""
    config = credentials.postgresql_connection()
    if force_fwd:
        config.db_ssh_forward = True
    with db(config) as db_conn:
        perf = db_conn.get_performance(experiment_name=experiment_name)
    return perf


def main(
        initialize_db):
    """Test the DB."""
    if initialize_db:
        print 'Initializing database.'
        initialize_database()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initialize",
        dest="initialize_db",
        action='store_true',
        help='Recreate your database.')
    args = parser.parse_args()
    main(**vars(args))
