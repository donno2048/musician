from                cv2                 import      imwrite
from               keras                import      backend
from            keras.layers            import      Activation, Dense, Dropout, Flatten, Input, Reshape, TimeDistributed
from     keras.layers.normalization     import      BatchNormalization
from            keras.losses            import      binary_crossentropy
from            keras.models            import      load_model, Model
from          keras.optimizers          import      Adam, RMSprop
from            keras.utils             import      plot_model
from             matplotlib             import      pyplot
from                mido                import      Message, MidiFile, MidiTrack
from               numpy                import      zeros, uint8, uint32, zeros_like, argmax, amax, maximum, where, save, copy, sum, arange, squeeze, mean, std, cov, sqrt, dot, array, expand_dims, load
from            numpy.linalg            import      svd
from            numpy.random            import      normal, seed as seen
from                 os                 import      mkdir, remove, walk
from              os.path               import      exists, join 
from               random               import      seed
def samples_to_midi(samples, file_name, threshold=0.5):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    ticks_per_sample = mid.ticks_per_beat / 24
    program_message = Message('program_change', program=1, time=0, channel=0)
    track.append(program_message)
    abs_time = 0
    last_time = 0
    for sample in samples:
        for y in range(sample.shape[0]):
            abs_time += ticks_per_sample
            for x in range(sample.shape[1]):
                note = x + 16
                if sample[y, x] >= threshold and (y == 0 or sample[y - 1, x] < threshold):
                    delta_time = abs_time - last_time
                    track.append(Message('note_on', note=int(note), velocity=127, time=int(delta_time)))
                    last_time = abs_time
                if sample[y, x] >= threshold and (y == sample.shape[0] - 1 or sample[y + 1, x] < threshold):
                    delta_time = abs_time - last_time
                    track.append(Message('note_off', note=int(note), velocity=127, time=int(delta_time)))
                    last_time = abs_time
    mid.save(file_name)
def generate_normalized_random_songs(x_orig, y_orig, encoder, decoder, random_vectors, write_dir):
    latent_x = squeeze(encoder.predict(y_orig))
    latent_mean = mean(latent_x, axis=0)
    latent_stds = std(latent_x, axis=0)
    latent_cov = cov((latent_x - latent_mean).T)
    _, pca_values, pca_vectors = svd(latent_cov)
    pca_values = sqrt(pca_values)
    save(write_dir + 'latent_means.npy', latent_mean)
    save(write_dir + 'latent_stds.npy', latent_stds)
    save(write_dir + 'latent_pca_values.npy', pca_values)
    save(write_dir + 'latent_pca_vectors.npy', pca_vectors)
    latent_vectors = latent_mean + dot(random_vectors * pca_values, pca_vectors)
    for i in range(latent_vectors.shape[0]): samples_to_midi(decoder([latent_vectors[i:i + 1], 0])[0][0], write_dir + 'random_vectors' + str(i) + '.mid', 32)
    title = ''
    if '\\' in write_dir: title = 'Epoch: ' + write_dir.split('\\')[-2][1:]
    pyplot.clf()
    pca_values[::-1].sort()
    pyplot.title(title)
    pyplot.bar(arange(pca_values.shape[0]), pca_values, align='center')
    pyplot.draw()
    pyplot.savefig(write_dir + 'latent_pca_values.png')
    pyplot.clf()
    pyplot.title(title)
    pyplot.bar(arange(pca_values.shape[0]), latent_mean, align='center')
    pyplot.draw()
    pyplot.savefig(write_dir + 'latent_means.png')
    pyplot.clf()
    pyplot.title(title)
    pyplot.bar(arange(pca_values.shape[0]), latent_stds, align='center')
    pyplot.draw()
    pyplot.savefig(write_dir + 'latent_stds.png')
def main():
    backend.set_image_data_format('channels_first')
    seen(0)
    seed(0)
    all_samples = []
    all_lengths = []
    for root, _, files in walk(".\\data"):
        for file in files:
            try:
                has_time_sig = False
                mid = MidiFile(join(root, file))
                ticks_per_beat = mid.ticks_per_beat
                ticks_per_measure = 4 * ticks_per_beat
                for msg in [msg for track in mid.tracks for msg in track if msg.type == 'time_signature']:
                    new_tpm = ticks_per_measure * msg.numerator / msg.denominator
                    if has_time_sig and new_tpm != ticks_per_measure: raise NotImplementedError('Multiple time signatures not supported')
                    ticks_per_measure = new_tpm
                    has_time_sig = True
                all_notes = {}
                for track in mid.tracks:
                    abs_time = 0
                    for msg in track:
                        abs_time += msg.time
                        if msg.type == 'program_change' and msg.program >= 112: break
                        if msg.type == 'note_on':
                            if not msg.velocity: continue
                            note = msg.note - 16
                            if note < 0 or note >= 96: continue
                            if note not in all_notes: all_notes[note] = []
                            else:
                                single_note = all_notes[note][-1]
                                if len(single_note) == 1: single_note.append(single_note[0] + 1)
                            all_notes[note].append([abs_time * 96 / ticks_per_measure])
                        elif msg.type == 'note_off' and len(all_notes[note][-1]) == 1: all_notes[note][-1].append(abs_time * 96 / ticks_per_measure)
                for note in all_notes:
                    for start_end in all_notes[note]:
                        if len(start_end) == 1:
                            start_end.append(start_end[0] + 1)
                samples = []
                for note in all_notes:
                    for start, end in all_notes[note]:
                        sample_ix = int(start / 96)
                        assert sample_ix < 2 ** 20
                        while len(samples) <= sample_ix: samples.append(zeros((96, 96), dtype=uint8))
                        sample = samples[sample_ix]
                        start_ix = int(start - sample_ix * 96)
                        sample[start_ix, int(note)] = 1
            except Exception as e:
                print("ERROR ", file)
                print(e)
                continue
            num_notes = samples[0].shape[1]
            merged_sample = zeros_like(samples[0])
            for sample in samples: merged_sample = maximum(merged_sample, sample)
            merged_sample = amax(merged_sample, axis=0)
            min_note = argmax(merged_sample)
            max_note = merged_sample.shape[0] - argmax(merged_sample[::-1])
            center_deviation = num_notes / 2 - (max_note + min_note) / 2
            out_samples = samples
            out_lengths = [len(samples), len(samples)]
            for i in range(len(samples)):
                out_sample = zeros_like(samples[i])
                out_sample[:, min_note + int(center_deviation):max_note + int(center_deviation)] = samples[i][:, min_note:max_note]
                out_samples.append(out_sample)
            all_samples += out_samples
            all_lengths += out_lengths
    for i in ['output', 'results', 'results\\history']:
        if not exists(i): mkdir(i)
    y_samples = array(all_samples, dtype=uint8)
    y_lengths = array(all_lengths, dtype=uint32)
    samples_qty = y_samples.shape[0]
    songs_qty = y_lengths.shape[0]
    assert (sum(y_lengths) == samples_qty)
    x_shape = (songs_qty, 1)
    x_orig = expand_dims(arange(x_shape[0]), axis=-1)
    y_shape = (songs_qty, 16) + y_samples.shape[1:]
    y_orig = zeros(y_shape, dtype=y_samples.dtype)
    song_start_ix = 0
    song_end_ix = y_lengths[0]
    for song_ix in range(songs_qty):
        ix = song_ix
        song_end_ix = song_start_ix + y_lengths[song_ix]
        for window_ix in range(16):
            song_measure_ix = (window_ix) % y_lengths[song_ix]
            y_orig[ix, window_ix] = y_samples[song_start_ix + song_measure_ix]
        song_start_ix = song_end_ix
    assert (song_end_ix == samples_qty)
    x_train = copy(x_orig)
    y_train = copy(y_orig)
    test_ix = 0
    y_test_song = copy(y_train[test_ix: test_ix + 1])
    x_test_song = copy(x_train[test_ix: test_ix + 1])
    samples_to_midi(y_test_song[0], '.\\output\\gt.mid')
    input_shape = y_shape[1:]
    x_in = Input(shape=input_shape)
    x = Reshape((input_shape[0], -1))(x_in)
    x = TimeDistributed(Dense(2000, activation='relu'))(x)
    x = TimeDistributed(Dense(200, activation='relu'))(x)
    x = Flatten()(x)
    x = Dense(1600, activation='relu')(x)
    x = Dense(120)(x)
    x = BatchNormalization(momentum=0.9, name='encoder')(x)
    x = Dense(1600, name='decoder')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(3200)(x)
    x = Reshape((16, 200))(x)
    x = TimeDistributed(BatchNormalization(momentum=0.9))(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    x = TimeDistributed(Dense(2000))(x)
    x = TimeDistributed(BatchNormalization(momentum=0.9))(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    x = TimeDistributed(Dense(input_shape[1] * input_shape[2], activation='sigmoid'))(x)
    x = Reshape((input_shape[0], input_shape[1], input_shape[2]))(x)
    model = Model(x_in, x)
    model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy')
    try: plot_model(model, to_file='.\\results\\model.png', show_shapes=True)
    except: pass
    decoder = backend.function([model.get_layer('decoder').input, backend.learning_phase()], [model.layers[-1].output])
    encoder = Model(inputs=model.input, outputs=model.get_layer('encoder').output)
    random_vectors = normal(0.0, 1.0, (10, 120))
    save('.\\output\\random_vectors.npy', random_vectors)
    train_loss = []
    offset = 0
    for epoch in range(2000): # trainings
        song_start_ix = 0
        for song_ix in range(songs_qty):
            song_end_ix = song_start_ix + y_lengths[song_ix]
            for window_ix in range(16):
                song_measure_ix = (window_ix + offset) % y_lengths[song_ix]
                y_train[song_ix, window_ix] = y_samples[song_start_ix + song_measure_ix]
            song_start_ix = song_end_ix
        assert (song_end_ix == samples_qty)
        offset += 1
        history = model.fit(y_train, y_train, batch_size=350, epochs=1)
        loss = history.history["loss"][-1]
        train_loss.append(loss)
        pyplot.clf()
        ax = pyplot.gca()
        ax.yaxis.tick_right()
        ax.yaxis.set_ticks_position('both')
        ax.yaxis.grid(True)
        pyplot.plot(train_loss)
        pyplot.ylim([0.0, 0.009])
        pyplot.xlabel('Epoch')
        pyplot.draw()
        pyplot.savefig('.\\results\\history\\losses.png')
        save_epoch = epoch + 1
    write_dir = '.\\results\\history\\e' + str(save_epoch)
    if not exists(write_dir): mkdir(write_dir)
    write_dir += '\\'
    model.save('.\\results\\history\\model.h5')
    y_song = model.predict(y_test_song, batch_size=350)[0]
    if not exists(write_dir + 'test'): mkdir(write_dir + 'test')
    for i in range(y_song.shape[0]): imwrite(write_dir + 'test' + '\\s' + str(i) + '.png', (1.0 - y_song[i]) * 255)
    samples_to_midi(y_song, write_dir + 'test.mid')
    generate_normalized_random_songs(x_orig, y_orig, encoder, decoder, random_vectors, write_dir)
if __name__ == '__main__':
    if exists('.\\data\\README'):remove('.\\data\\README')
    main()
