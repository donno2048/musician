from cv2 import imwrite
from tensorflow.keras import backend
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, Input, Reshape, TimeDistributed
from tensorflow.python.keras.layers.normalization import BatchNormalization
from keras.losses import binary_crossentropy
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from matplotlib import pyplot
from mido import Message, MidiFile, MidiTrack
from numpy import transpose, minimum, rot90, full, clip, int16, abs, sin, mod, sign, zeros, uint8, uint32, float32, zeros_like, argmax, amax, maximum, where, save, copy, sum, arange, squeeze, mean, std, cov, sqrt, dot, array, expand_dims, load, exp
from numpy.linalg import svd
from numpy.random import normal, seed as seen
from os import mkdir, remove, walk
from os.path import exists, join
from random import seed
from math import pi
from wave import open
from pyaudio import PyAudio, paContinue
TRAIN=2000
note_dt = 2000
note_threshold = 32
prev_mouse_pos = None
mouse_pressed = 0
cur_slider_ix = 0
cur_control_ix = 0
volume = 3000
instrument = 0
needs_update = True
current_params = zeros((120), dtype=float32)
current_notes = zeros((16, 96, 96), dtype=uint8)
cur_controls = array([0.75, 0.5, 0.5], dtype=float32)
audio = PyAudio()
audio_notes = []
audio_time = 0
note_time = 0
note_time_dt = 0
audio_reset = False
audio_pause = False
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
def setup():
    if exists('.\\data\\README'): remove('.\\data\\README')
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
                print("ERROR ", file, e)
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
    for epoch in range(TRAIN):
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
    write_dir = '.\\results\\history\\e' + str(TRAIN)
    if not exists(write_dir): mkdir(write_dir)
    write_dir += '\\'
    model.save('.\\results\\history\\model.h5')
    y_song = model.predict(y_test_song, batch_size=350)[0]
    if not exists(write_dir + 'test'): mkdir(write_dir + 'test')
    for i in range(y_song.shape[0]): imwrite(write_dir + 'test' + '\\s' + str(i) + '.png', (1.0 - y_song[i]) * 255)
    samples_to_midi(y_song, write_dir + 'test.mid')
    generate_normalized_random_songs(x_orig, y_orig, encoder, decoder, random_vectors, write_dir)
def audio_callback(in_data, frame_count, time_info, status):
    global audio_time, audio_notes, audio_reset, note_time, note_time_dt
    if audio_reset:
        audio_notes = []
        audio_time = 0
        note_time = 0
        note_time_dt = 0
        audio_reset = False
    if audio_pause and status is not None: return zeros((frame_count,), dtype=float32).tobytes(), paContinue
    while note_time_dt < audio_time + frame_count:
        if note_time >= 1536: break
        for note in where(current_notes[note_time // 96, note_time % 96] >= note_threshold)[0]:
            audio_notes.append((note_time_dt, 3889 * pow(2.0, note / 12.0) / 2400000))
        note_time += 1
        note_time_dt += note_dt
    data = zeros((frame_count,), dtype=float32)
    for t, f in audio_notes:
        x = maximum(arange(audio_time - t, audio_time + frame_count - t), 0)
        if instrument == 0: w = sign(1 - mod(x * f, 2))
        elif instrument == 1: w = mod(x * f - 1, 2) - 1
        elif instrument == 2: w = 2 * abs(mod(x * f - 0.5, 2) - 1) - 1
        elif instrument == 3: w = sin(x * f * pi)
        w[x == 0] = 0
        w *= volume * exp(-x / 9600)
        data += w
    data = clip(data, -32000, 32000).astype(int16)
    audio_time += frame_count
    audio_notes = [(t, f) for t, f in audio_notes if audio_time < t + 20000]
    if note_time / 96 >= 16:
        audio_time = 0
        note_time = 0
        note_time_dt = 0
        audio_notes = []
    return data.tobytes(), paContinue
def get_pianoroll_from_notes(notes):
    output = full((3, 200, 800), 64, dtype=uint8)
    for i in range(2):
        for j in range(8):
            measure = rot90(notes[i * 8 + j])
            played_only = where(measure >= note_threshold, 255, 0)
            output[0, 2 + i * 100:98 + i * 100, 2 + j * 100:98 + j * 100] = minimum(measure * (255.0 / note_threshold), 255.0)
            output[1, 2 + i * 100:98 + i * 100, 2 + j * 100:98 + j * 100] = played_only
            output[2, 2 + i * 100:98 + i * 100, 2 + j * 100:98 + j * 100] = played_only
    return transpose(output, (2, 1, 0))
def main():
    import pygame
    if not exists(f'results\\history\\e{TRAIN}'): setup()
    global mouse_pressed, current_notes, audio_pause, needs_update, current_params, prev_mouse_pos, audio_reset, instrument, cur_slider_ix, cur_control_ix, note_threshold, note_dt, volume
    backend.set_image_data_format('channels_first')
    model = load_model('results\\history\\model.h5')
    encoder = Model(inputs=model.input, outputs=model.get_layer('encoder').output)
    decoder = backend.function([model.get_layer('decoder').input, backend.learning_phase()], [model.layers[-1].output])
    latent_means = load(f'results\\history\\e{TRAIN}\\latent_means.npy')
    latent_stds = load(f'results\\history\\e{TRAIN}\\latent_stds.npy')
    latent_pca_values = load(f'results\\history\\e{TRAIN}\\latent_pca_values.npy')
    latent_pca_vectors = load(f'results\\history\\e{TRAIN}\\latent_pca_vectors.npy')
    y_samples = load('.\\output\\samples.npy')
    y_lengths = load('.\\output\\lengths.npy')
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((800, 440))
    notes_surface = screen.subsurface((0, 210, 800, 200))
    pygame.display.set_caption('Neural Composer')
    audio_stream = audio.open(format=audio.get_format_from_width(2), channels=1, rate=48000, output=True, stream_callback=audio_callback)
    audio_stream.start_stream()
    running = True
    random_song_ix = 0
    cur_len = 0
    note_threshold = 210 - cur_controls[0] * 200
    note_dt = 2000 - cur_controls[1] * 1800
    volume = cur_controls[2] * 6000
    while running:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if pygame.mouse.get_pressed()[0]:
                    prev_mouse_pos = pygame.mouse.get_pos()
                    if 5 <= prev_mouse_pos[0] < 805 and 5 <= prev_mouse_pos[1] < 215:
                        cur_slider_ix = (prev_mouse_pos[0] - 5) // 19
                        mouse_pressed = 1
                    if 85 <= prev_mouse_pos[0] < 715 and 410 <= prev_mouse_pos[1] < 440:
                        cur_control_ix = (prev_mouse_pos[0] - 85) // 210
                        mouse_pressed = 2
                    if mouse_pressed == 1 and 5 <= prev_mouse_pos[1] <= 205:
                        current_params[int(cur_slider_ix)] = prev_mouse_pos[1] / 20 - 5.25
                        needs_update = True
                    elif mouse_pressed == 2 and 90 <= prev_mouse_pos[0] - cur_control_ix * 210 <= 290:
                        cur_controls[int(cur_control_ix)] = (prev_mouse_pos[0] - 90 - cur_control_ix * 210) / 200
                        note_threshold = 210 - cur_controls[0] * 200
                        note_dt = 2000 - cur_controls[1] * 1800
                        volume = cur_controls[2] * 6000
                elif pygame.mouse.get_pressed()[2]:
                    current_params = zeros((120), dtype=float32)
                    needs_update = True
            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_pressed = 0
                prev_mouse_pos = None
            elif event.type == pygame.MOUSEMOTION and mouse_pressed > 0:
                if mouse_pressed == 1 and 5 <= pygame.mouse.get_pos()[1] <= 205:
                    current_params[int(cur_slider_ix)] = pygame.mouse.get_pos()[1] / 20 - 5.25
                    needs_update = True
                elif mouse_pressed == 2 and 90 <= pygame.mouse.get_pos()[0] - cur_control_ix * 210 <= 290:
                    cur_controls[int(cur_control_ix)] = (pygame.mouse.get_pos()[0] - 90 - cur_control_ix * 210) / 200
                    note_threshold = 210 - cur_controls[0] * 200
                    note_dt = 2000 - cur_controls[1] * 1800
                    volume = cur_controls[2] * 6000
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_r, pygame.K_e]:
                    current_params = clip(normal(0.0, 2.0 if event.key == pygame.K_e else 1.0, (120)), -5, 5)
                    needs_update = True
                    audio_reset = True
                if event.key == pygame.K_o:
                    current_notes = y_samples[cur_len:cur_len + 16] * 255
                    latent_x = encoder.predict(expand_dims(y_samples[cur_len:cur_len + 16], 0), batch_size=1)[0]
                    cur_len += y_lengths[random_song_ix]
                    random_song_ix += 1
                    current_params = dot(latent_x - latent_means, latent_pca_vectors.T) / latent_pca_values
                    needs_update = True
                    audio_reset = True
                if event.key == pygame.K_m:
                    audio_pause = True
                    audio_reset = True
                    samples_to_midi(current_notes, 'results/live.mid', note_threshold)
                    audio_pause = False
                if event.key == pygame.K_w:
                    audio_pause = True
                    audio_reset = True
                    save_audio = b''
                    while True:
                        save_audio += audio_callback(None, 1024, None, None)[0]
                        if not audio_time: break
                    wave_output = open('results/live.wav', 'w')
                    wave_output.setparams((1, 2, 48000, 0, 'NONE', 'not compressed'))
                    wave_output.writeframes(save_audio)
                    wave_output.close()
                    audio_pause = False
                if event.key in [pygame.K_ESCAPE, pygame.Quit]:
                    running = False
                    break
                if event.key == pygame.K_SPACE: audio_pause = not audio_pause
                if event.key == pygame.K_TAB: audio_reset = True
                if event.key == pygame.K_1: instrument = 0
                if event.key == pygame.K_2: instrument = 1
                if event.key == pygame.K_3: instrument = 2
                if event.key == pygame.K_4: instrument = 3
                if event.key == pygame.K_c:
                    current_params = dot(encoder.predict(expand_dims(where(current_notes > note_threshold, 1, 0), 0))[0] - latent_means, latent_pca_vectors.T) / latent_pca_values
                    needs_update = True
        if needs_update:
            current_notes = (decoder([expand_dims(latent_means + dot(current_params * latent_pca_values, latent_pca_vectors), axis=0), 0])[0][0] * 255.0).astype(uint8)
            needs_update = False
        screen.fill((210, 210, 210))
        pygame.surfarray.blit_array(notes_surface, get_pianoroll_from_notes(current_notes))
        pygame.draw.rect(screen, (255, 255, 0), (2 + (note_time // 96 % 8) * 100 + note_time % 96, 212 + (note_time // 768) * 100, 4, 96), 0)
        for i in range(40):
            slider_color = [(90, 20, 20), (90, 90, 20), (20, 90, 20), (20, 90, 90), (20, 20, 90), (90, 20, 90)][i % 6]
            pygame.draw.line(screen, slider_color, (14.5 + i * 19, 5), (14.5 + i * 19, 205))
            for j in range(11): pygame.draw.line(screen, (0, 0, 0) if j == 5 else slider_color, (9 + i * 19, 5 + j * 20), (20 + i * 19, 5 + j * 20))
            pygame.draw.circle(screen, slider_color, (14 + i * 19, int(20 * current_params[i] + 105)), 7)
        for i in range(3):
            pygame.draw.rect(screen, [(255, 0, 0), (0, 255, 0), (0, 0, 255)][i], (90 + i * 210, 415, int(200 * cur_controls[i]), 20))
            pygame.draw.rect(screen, (0, 0, 0), (90 + i * 210, 415, 200, 20), 1)
        pygame.display.flip()
        pygame.time.wait(10)
    audio_stream.stop_stream()
    audio_stream.close()
    audio.terminate()
if __name__ == '__main__': main()
