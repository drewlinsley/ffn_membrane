import sys, getopt
import tensorflow as tf


usage_str = 'python fix_membrane_ckpt.py --checkpoint_dir=path/to/dir/ output_dir=path/to/dir/ --dry_run'


def reshape(checkpoint_dir, dry_run, output_dir, keyword='out_embedding', fan_out=3):
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)
            if keyword in var_name:
                # Set the new shape
                var = var[..., :fan_out]
                if dry_run:
                    print('%s would be have fan_out: %s.' % (var_name, fan_out))
                else:
                    print('Changing fan_out for %s to %s.' % (var_name, fan_out))
                    # Rename the variable
                    var = tf.Variable(var, name=var_name)
            else:
                print('%s remains unchanged' % var_name)
                var = tf.Variable(var, name=var_name)

        if not dry_run:
            # Save the variables
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            # saver.save(sess, checkpoint.model_checkpoint_path)
            saver.save(sess, output_dir)


def main(argv):
    checkpoint_dir = None
    replace_from = None
    replace_to = None
    add_prefix = None
    dry_run = False

    try:
        opts, args = getopt.getopt(argv, 'h', ['help=', 'checkpoint_dir=',
                                               'output_dir=', 'dry_run'])
    except getopt.GetoptError:
        print(usage_str)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(usage_str)
            sys.exit()
        elif opt == '--checkpoint_dir':
            checkpoint_dir = arg
        elif opt == '--output_dir':
            output_dir = arg
        elif opt == '--dry_run':
            dry_run = True

    if not checkpoint_dir:
        print('Please specify a checkpoint_dir. Usage:')
        print(usage_str)
        sys.exit(2)

    reshape(checkpoint_dir=checkpoint_dir, output_dir=output_dir, dry_run=dry_run)


if __name__ == '__main__':
    main(sys.argv[1:])

