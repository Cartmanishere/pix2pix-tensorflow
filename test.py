import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from datetime import datetime
from model import *

class ModelTest:

    def test_with_images(self):
        if a.seed is None:
            a.seed = random.randint(0, 2**31 - 1)

        tf.set_random_seed(a.seed)
        np.random.seed(a.seed)
        random.seed(a.seed)

        if not os.path.exists(a.output_dir):
            os.makedirs(a.output_dir)

        self.examples = load_examples(os.path.join('test_data', 'inputs'))
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
            else:
                print('[!] Checkpoint not found')
                exit(1)


            max_steps = self.examples.steps_per_epoch

            print("[*] Starting Testing")

            for step in range(max_steps):
                t1 = time.time()
                results = sess.run(self.display_fetches)
                save_images(results, step=step, output_dir=a.test_dir)

                print('[*] Time taken - %0.3fs' % (time.time() - t1))


    def test_with_camera(self, cap):

        name = 'cv2'
        try:
            cv2 = __import__(name)
        except ImportError as e:
            print(e)
            print('[!] Need CV2 module for this mode')
            exit(1)

        cap = cv2.VideoCapture(cap)

        target_tf = tf.placeholder(tf.float32, shape=[None, None, 3]) # this is not used because we're only generating

        img_contents = tf.placeholder(dtype=tf.string)
        img_decoded = tf.image.decode_png(img_contents)
        image = tf.image.convert_image_dtype(img_decoded, dtype=tf.float32)
        image.set_shape([None, None, 3])
        image = preprocess(image)

        batch_input = tf.expand_dims(image, axis=0)
        batch_target = tf.expand_dims(target_tf, axis=0)

        model = create_model(batch_input, batch_target)
        outputs = deprocess(model.outputs)

        def convert(image):
            if a.aspect_ratio != 1.0:
                # upscale to correct aspect ratio
                size = [a.CROP_SIZE, int(round(a.CROP_SIZE * a.aspect_ratio))]
                image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

            return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)


        with tf.name_scope("convert_outputs"):
            converted_outputs = convert(outputs)

        saver = tf.train.Saver(max_to_keep=1)
        sv = tf.train.Supervisor(logdir=a.checkpoint, save_summaries_secs=0, saver=None)

        with sv.managed_session() as sess:

            checkpoint_dir = get_checkpoint()
            if checkpoint_dir is not None and tf.train.checkpoint_exists(checkpoint_dir):
                print("[*] Loading model from checkpoint")
                t1 = time.time()
                checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
                saver.restore(sess, checkpoint)
                print('[*] Time Taken to load checkpoints - {:.3f} s'.format(time.time()-t1 ))
            else:
                print('[!] Checkpoint not found')
                exit(1)

            frame_w = a.scale_size
            frame_h = 2 * a.scale_size

            while True:
                
                t1 = time.time()
                ret, test_image = cap.read()

                test_image_ = cv2.resize(test_image, (a.CROP_SIZE, a.CROP_SIZE))
                test_image = cv2.cvtColor(test_image_, cv2.COLOR_BGR2RGB)

                img = Image.fromarray(test_image)
                imgByteArr = io.BytesIO()
                img.save(imgByteArr, format='PNG')
                imgByteArr = imgByteArr.getvalue()

                max_steps = 1
                for step in range(max_steps):

                    results = sess.run(converted_outputs, feed_dict={img_contents: imgByteArr, target_tf: test_image})


                results[0] = cv2.cvtColor(results[0], cv2.COLOR_RGB2BGR)

                cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                
                cv2.putText(results[0], '%s' % (datetime.now().strftime('%Hh:%Mm:%Ss')), (10, 246), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (18, 156, 243), 2)
                
                res = np.hstack((test_image_, results[0])).astype(np.uint8)
                
                cv2.imshow('window', results[0])
    
                print('[*] Time taken - %0.3fs' % (time.time() - t1))
                opt = cv2.waitKey(1)
                if opt & 0xFF == ord('q'):
                    break
                

            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()



if __name__ == "__main__":
    model = ModelTest()
    if a.test_mode == 'stream':
        if a.stream_url == 'local':
            cap = 0
        else:
            cap = args.stream_url

        model.test_with_camera(cap)

    else:
        model.test_with_images()
