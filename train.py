import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from model import *

class ModelTrain:

    def build_graph(self):
        if a.seed is None:
            a.seed = random.randint(0, 2**31 - 1)

        tf.set_random_seed(a.seed)
        np.random.seed(a.seed)
        random.seed(a.seed)

        if not os.path.exists(a.output_dir):
            os.makedirs(a.output_dir)

        self.examples = load_examples(a.train_dir)
        self.model = create_model(self.examples.inputs, self.examples.targets)

        inputs = deprocess(self.examples.inputs)
        targets = deprocess(self.examples.targets)
        outputs = deprocess(self.model.outputs)

        with tf.name_scope("convert_inputs"):
            converted_inputs = convert(inputs)

        with tf.name_scope("convert_targets"):
            converted_targets = convert(targets)

        with tf.name_scope("convert_outputs"):
            converted_outputs = convert(outputs)

        with tf.name_scope("encode_images"):
            self.display_fetches = {
                "paths": self.examples.paths,
                "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
                "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
                "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
            }

        # summaries
        with tf.name_scope("inputs_summary"):
            tf.summary.image("inputs", converted_inputs)

        with tf.name_scope("targets_summary"):
            tf.summary.image("targets", converted_targets)

        with tf.name_scope("outputs_summary"):
            tf.summary.image("outputs", converted_outputs)

        with tf.name_scope("predict_real_summary"):
            tf.summary.image("predict_real", tf.image.convert_image_dtype(self.model.predict_real, dtype=tf.uint8))

        with tf.name_scope("predict_fake_summary"):
            tf.summary.image("predict_fake", tf.image.convert_image_dtype(self.model.predict_fake, dtype=tf.uint8))

        tf.summary.scalar("discriminator_loss", self.model.discrim_loss)
        tf.summary.scalar("generator_loss_GAN", self.model.gen_loss_GAN)
        tf.summary.scalar("generator_loss_L1", self.model.gen_loss_L1)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name + "/values", var)

        for grad, var in self.model.discrim_grads_and_vars + self.model.gen_grads_and_vars:
            tf.summary.histogram(var.op.name + "/gradients", grad)

        with tf.name_scope("parameter_count"):
            self.parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    def train_model(self):
        saver = tf.train.Saver(max_to_keep=1)
        logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
        sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
        with sv.managed_session() as sess:
            print("[*] Parameter Count - {}".format(sess.run(self.parameter_count)))

            checkpoint_dir = get_checkpoint()
            if checkpoint_dir is not None and tf.train.checkpoint_exists(checkpoint_dir):
                print("[*] Loading model from checkpoint")
                t1 = time.time()
                checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
                saver.restore(sess, checkpoint)
                print('[*] Time Taken to load checkpoints - {:.3f} s'.format(time.time()-t1 ))


            max_steps = 2**32
            if a.max_epochs is not None:
                max_steps = self.examples.steps_per_epoch * a.max_epochs
            if a.max_steps is not None:
                max_steps = a.max_steps

            print("[*] Starting training")
            start = time.time()

            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": self.model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):
                    fetches["discrim_loss"] = self.model.discrim_loss
                    fetches["gen_loss_GAN"] = self.model.gen_loss_GAN
                    fetches["gen_loss_L1"] = self.model.gen_loss_L1

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = self.display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(a.summary_freq):
                    print("[*] Recording Summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):
                    print("[*] Saving Display Images")
                    filesets = save_images(results["display"], step=results["global_step"])
                    append_index(filesets, step=True)

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / self.examples.steps_per_epoch)
                    max_epochs = math.ceil(max_steps / self.examples.steps_per_epoch)

                    epoch_info = '[%d/%d Epochs]' % (train_epoch, max_epochs)
                    train_step = (results["global_step"] - 1) % self.examples.steps_per_epoch + 1

                    step_info = '[%d/%d Steps]' % (train_step, self.examples.steps_per_epoch )

                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate

                    progress_info = "%0.2fimg/sec - %0.2fm" % (rate, remaining / 60)
                    loss_info = "d_loss = %.5f gen_loss_GAN %.5f gen_loss_L1 %.5f" % (results['discrim_loss'], results['gen_loss_GAN'], results['gen_loss_L1'])

                    print("%15s %10s %55s %s" % (epoch_info, step_info, loss_info, progress_info))

                if should(a.save_freq):
                    print("[*] Saving Model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break


if __name__ == "__main__":
    model = ModelTrain()
    model.build_graph()
    model.train_model()
