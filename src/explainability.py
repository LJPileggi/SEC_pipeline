import os
import numpy as np
import scipy
from scipy import signal
import torch
import torch.nn.functional as F
import librosa
import librosa.display # Often useful for visualization

from .utils import epochs, batch_size, device, sampling_rate, ref, center_freqs
from .models import CLAP_initializer, spectrogram_n_octaveband_generator, FinetunedModel
from .losses import build_optimizer

### TODOs list: ###

### listenable_wav_from_n_octaveband ###
# set correct directory to save explanations from listenable_wav_from_n_octaveband

### LMAC ###
### generate_listenable_interpretation ###
# set correct reconstructed audio path for generate_listenable_interpretations

### LMAC_explainer ###
# finish fixing and honing the pipeline
# correctly set up multi-GPU mode
# finish writing comments

###  ###

### LMAC ###

### Explainability models ###

def listenable_wav_from_n_octaveband(my_3octave_spectrogram_dB, track_name, sampling_rate):
    # Assume you have 'wav_data' and 'sampling_rate' from your original context

    # --- Your existing spectrogram calculation (output is in dB) ---
    # def spectrogram_3octaveband_generator(...):
        # ... (your original code) ...
        # return spectrogram.T # Ensure (time_frames, freq_bands)

    # Example Usage (assuming you have a 'dummy_wav' loaded)
    # dummy_wav = np.random.randn(sampling_rate * 10) # 10 seconds of dummy audio
    # my_3octave_spectrogram_dB = spectrogram_3octaveband_generator(
    #     wav_data=dummy_wav, sampling_rate=sampling_rate
    # )

    # --- Step 1: Convert your 3-octave band spectrogram (in dB) to linear amplitude ---
    # Ensure your spectogram is (freq_bands, time_frames) if librosa expects it like that
    # (your function returns (time_frames, freq_bands), so we need to adjust)
    linear_amplitude_spectrogram_3octave = ref * (10 ** (my_3octave_spectrogram_dB.T / 20))

    # --- Step 2: Adapt to an STFT-like magnitude spectrogram ---
    # This is the tricky and lossy part: mapping 3-octave bands to STFT bins.
    # You'll need to define STFT parameters.
    n_fft = 2048
    hop_length = 512 # Or another suitable value

    # Create a mapping from 3-octave bands to STFT frequency bins.
    # This is a conceptual placeholder. You might need a more sophisticated mapping
    # (e.g., weighting bins, interpolating).
    # For simplicity, let's assume we can linearly interpolate or just assign
    # the 3-octave band value to a range of STFT bins.
    # This step is highly approximate and will significantly impact reconstruction quality.

    # Example of a very basic, naive mapping (you'd need a more robust one)
    stft_freqs = librosa.fft_frequencies(sr=sampling_rate, n_fft=n_fft)
    # Find closest STFT bin index for each 3-octave band center frequency
    stft_magnitudes_approx = np.zeros((n_fft // 2 + 1, linear_amplitude_spectrogram_3octave.shape[1]))

    for i, cf in enumerate(center_freqs):
        # Find the closest STFT frequency bin to the center frequency of the 3-octave band
        closest_stft_bin_idx = np.argmin(np.abs(stft_freqs - cf))
        # Assign the 3-octave band amplitude to this STFT bin
        stft_magnitudes_approx[closest_stft_bin_idx, :] = linear_amplitude_spectrogram_3octave[i, :]

    # This `stft_magnitudes_approx` will be sparse and not ideal.
    # A better approach might involve spreading the energy across the STFT bins
    # that fall within the 3-octave band's range. This is an advanced topic.

    # --- Step 3: Apply Griffin-Lim ---
    # Note: Griffin-Lim expects magnitude spectrograms (non-negative, real numbers).
    # The output is a time-domain audio waveform.
    reconstructed_audio_approx = librosa.griffinlim(
        S=stft_magnitudes_approx, # Your approximate STFT magnitude spectrogram
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft, # Often n_fft for window length
        n_iter=32 # Number of iterations for convergence (higher = better, but slower)
    )

    # You can then save or play reconstructed_audio_approx
    # TODO: set correct directory to save explanations from listenable_wav_from_n_octaveband
    reconstructed_name = "{track_name.replace('.pt', '')}_reconstructed.wav"
    librosa.output.write(os.path.join(basedir, reconstructed_name), reconstructed_audio_approx, sampling_rate)

    print(f"Shape of approximated reconstructed audio: {reconstructed_audio_approx.shape}")

# --- PLACEHOLDER PER VARIABILI GLOBALI O DA PASSARE ---
# Queste variabili devono essere definite nel tuo ambiente prima di usare le classi.
# Ad esempio, dal tuo setup di generazione dello spettrogramma 3-ottava.
# num_octave_bands = 27 # Esempio: il numero di bande d'ottava che hai
# ref = 1.0 # Valore di riferimento per la conversione dB a lineare
# Esempio di center_freqs (devi usare quelle effettive dal tuo generatore di bande d'ottava)
# center_freqs = librosa.mel_frequencies(n_mels=num_octave_bands, fmin=20, fmax=20000) # Questa è una semplificazione, le bande d'ottava sono diverse dai mel.
# In un'applicazione reale, useresti le frequenze centrali precise delle tue bande d'ottava.
# Ad esempio: from py_sound_events.rir_util import octave_band_filter_bank
# fb = octave_band_filter_bank(sr=sampling_rate, N_fft=n_fft, bands_per_octave=3, f_min=20, f_max=20000)
# center_freqs = fb['center_freqs']

# Supponiamo che `clap_model` sia definito e importato altrove
# Come esempio, ho incluso una DummyCLAPModel nel secondo tab per testare.

# FinetunedModel (il tuo classificatore che prende embeddings 2D)

class Decoder(nn.Module):
    """
    Decoder network (M_theta) to generate the binary mask M.
    It takes latent representations 'h' (2D embeddings) from the classifier and
    generates a mask for the 3-octave band spectrogram.
    """
    def __init__(self, latent_dim_input, output_spectrogram_shape):
        super(Decoder, self).__init__()
        # output_spectrogram_shape expected (channels, num_octave_bands, time_frames)
        self.output_shape = output_spectrogram_shape

        self.initial_height = 8 # Esempio di dimensione spaziale iniziale
        self.initial_width = 8
        self.initial_channels = latent_dim_input // (self.initial_height * self.initial_width)

        if self.initial_channels == 0:
            # Aumenta le dimensioni iniziali se latent_dim_input è troppo grande per i canali
            # o se initial_height/width sono troppo grandi per i canali.
            # Questo è un valore di default che potrebbe dover essere adattato.
            self.initial_channels = 16 # Esempio, assicurati che latent_dim_input sia divisibile o >=
            self.initial_height = int(np.sqrt(latent_dim_input / self.initial_channels))
            self.initial_width = self.initial_height
            # Ricalcola se necessario:
            # if self.initial_height * self.initial_width * self.initial_channels != latent_dim_input:
            #     raise ValueError("Need a better strategy for initial_height/width/channels given latent_dim_input")

            # Per ora, in caso di 0, si può forzare un default o gestire l'errore
            if self.initial_height == 0 or self.initial_width == 0:
                 raise ValueError(f"Latent dim input {latent_dim_input} is too small to form initial 4D shape with current logic.")

        self.fc_to_4d = nn.Linear(latent_dim_input, self.initial_channels * self.initial_height * self.initial_width)

        # Layers de-convoluzionali. Assicurati che il numero di deconv layers
        # e i loro stride/padding/output_padding portino a una dimensione vicina
        # a (num_octave_bands, time_frames) dopo l'ultimo deconv layer.
        # Ad esempio, se partiamo da 8x8 e vogliamo arrivare a 27xN, avremo bisogno di:
        # 8 -> 16 -> 32 (stride=2 due volte, ma 27 è strano per i binari)
        # Sarà quasi sempre necessario un F.interpolate finale per l'adattamento preciso.

        self.deconv1 = nn.ConvTranspose2d(self.initial_channels, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        # L'ultimo deconv layer dovrebbe avere 1 canale in output per la maschera binaria
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.output_conv = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, h):
        x = self.fc_to_4d(h)
        x = x.view(-1, self.initial_channels, self.initial_height, self.initial_width)

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))

        m_raw = self.output_conv(x)

        # Target shape for interpolation: (num_octave_bands, time_frames)
        # output_spectrogram_shape è (batch, channels, num_octave_bands, time_frames)
        target_height = self.output_shape[-2] # num_octave_bands
        target_width = self.output_shape[-1]  # time_frames

        m_raw = m_raw.float()
        if m_raw.dim() == 3:
            m_raw = m_raw.unsqueeze(0)

        m_raw = F.interpolate(m_raw, size=(target_height, target_width), mode='bilinear', align_corners=False)

        mask = torch.sigmoid(m_raw)
        return mask

class LMAC:
    def __init__(self, classifier, decoder, clap_audio_encoder, alpha=1.0,beta=1.0,
                        lambda_in=1.0, lambda_out=1.0, lambda_g=0.0, lambda_s=0.0):
        self.classifier = classifier
        self.decoder = decoder
        self.clap_audio_encoder = clap_audio_encoder
        self.alpha = alpha
        self.beta = beta
        self.lambda_in = lambda_in
        self.lambda_out = lambda_out
        self.lambda_g = lambda_g
        self.lambda_s = lambda_s
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')

    # Nuova funzione per la conversione da spettrogramma a 3 bande d'ottava a waveform
    def     _spectrogram_n_octave_to_clap_waveform(self, masked_n_octave_spectrogram_linear
                        original_audio_shape, sampling_rate, track_name_suffix="masked"):
        """
        Converts a masked 3-octave band spectrogram (linear amplitude) to an audio waveform
        suitable for CLAP's audio encoder.
        This uses the Griffin-Lim algorithm.

        Args:
            masked_n_octave_spectrogram_linear (torch.Tensor): Masked spectrogram in linear amplitude.
                                                            Shape: (batch_size, 1, num_octave_bands, time_frames)
            original_audio_shape (torch.Size): Original audio waveform shape (batch_size, num_samples).
            sampling_rate (int): Sampling rate of the audio.
            track_name_suffix (str): Suffix for saving the reconstructed audio.
                                    Used for debugging/inspection.

        Returns:
            torch.Tensor: Reconstructed audio waveform, (batch_size, num_samples), for CLAP.
        """
        # Griffin-Lim opera su singoli spettrogrammi. Dobbiamo iterare sul batch.
        reconstructed_waveforms = []

        n_fft = 2048 # Parametri STFT per Griffin-Lim
        hop_length = 512

        # Assicurati che masked_n_octave_spectrogram_linear sia un tensore CPU per numpy/librosa
        # e sia in formato (num_octave_bands, time_frames) per la conversione.
        # Il tuo input è (batch_size, 1, num_octave_bands, time_frames).

        for i in range(masked_n_octave_spectrogram_linear.shape[0]):
            current_3octave_spec = masked_n_octave_spectrogram_linear[i, 0, :, :].detach().cpu().numpy()

            # --- Adatta a un STFT-like magnitude spectrogram (Step 2 dalla tua funzione) ---
            stft_freqs = librosa.fft_frequencies(sr=sampling_rate, n_fft=n_fft)
            stft_magnitudes_approx = np.zeros((n_fft // 2 + 1, current_3octave_spec.shape[1]), dtype=np.float32)

            for j, cf in enumerate(center_freqs):
                closest_stft_bin_idx = np.argmin(np.abs(stft_freqs - cf))
                # Assegna il valore della banda d'ottava all'indice del bin STFT.
                # Questa è una mappatura grezza, come hai notato.
                stft_magnitudes_approx[closest_stft_bin_idx, :] = current_3octave_spec[j, :]

            # --- Applica Griffin-Lim (Step 3 dalla tua funzione) ---
            reconstructed_audio_np = librosa.griffinlim(
                S=stft_magnitudes_approx,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                n_iter=32
            )

            # CLAP potrebbe avere una lunghezza fissa. Pad/tronca se necessario.
            target_samples = original_audio_shape[1]
            if reconstructed_audio_np.shape[0] < target_samples:
                pad_len = target_samples - reconstructed_audio_np.shape[0]
                reconstructed_audio_np = np.pad(reconstructed_audio_np, (0, pad_len))
            elif reconstructed_audio_np.shape[0] > target_samples:
                reconstructed_audio_np = reconstructed_audio_np[:target_samples]

            reconstructed_waveforms.append(torch.from_numpy(reconstructed_audio_np).float().to(masked_n_octave_spectrogram_linear.device))

        return torch.stack(reconstructed_waveforms) # Output: (batch_size, num_samples)

    def calculate_masking_loss(self, linear_n_octave_spectrogram_X, audio_waveforms, y_true_labels, sampling_rate):
        """
        Calculates the masking objective function for L-MAC.
        Args:
            linear_n_octave_spectrogram_X (torch.Tensor): 3-octave band spectrogram (linear amplitude) of original audio.
                                                Shape: (batch_size, 1, num_octave_bands, time_frames)
            audio_waveforms (torch.Tensor): Raw audio waveforms for CLAP processing.
                                            Shape: (batch_size, num_samples)
            y_true_labels (torch.Tensor): True class labels for the original audio.
                                            Shape: (batch_size,)
            sampling_rate (int): Sampling rate of the audio (needed for Griffin-Lim).
        """
        device = linear_n_octave_spectrogram_X.device

        # Step 1: Get latent representation 'h' and predicted class 'y_hat' from the original input
        self.clap_audio_encoder.eval() # Assicurati che l'encoder sia in modalità valutazione
        with torch.no_grad():
            preprocessed_audio_original = clap_model.preprocess_audio(audio_waveforms.tolist(), True)
            preprocessed_audio_original = preprocessed_audio_original.squeeze(1) # [batch, 1024]
            h_original = self.clap_audio_encoder(preprocessed_audio_original)[0][0]

            original_logits = self.classifier(h_original)
            y_hat = torch.argmax(original_logits, dim=1)

        # Step 2: Generate the binary mask M using the decoder
        # Il decoder prende l'embedding 2D h_original e genera una maschera 4D
        M = self.decoder(h_original.detach()) # M_theta(h)
        M = torch.clamp(M, 0, 1)

        # Step 3: Compute masked-in and masked-out inputs
        # Applica la maschera allo spettrogramma a 3 bande d'ottava (in ampiezza lineare)
        # Nota: my_3octave_spectrogram_dB.T è stato convertito a lineare.
        # linear_n_octave_spectrogram_X deve essere già in ampiezza lineare.

        # Converte dB a lineare se il tuo input è in dB
        # linear_amplitude_spectrogram_3octave = ref * (10 ** (linear_n_octave_spectrogram_X / 20))
        # No, se il tuo input è già "linear_n_octave_spectrogram_X", suppongo sia lineare.
        # Se invece è in dB, devi fare la conversione qui:
        # linear_n_octave_spectrogram_X_linear = ref * (10 ** (linear_n_octave_spectrogram_X / 20))
        # Per semplicità, assumo che `linear_n_octave_spectrogram_X` sia già in scala lineare.

        X_masked_in = M * linear_n_octave_spectrogram_X
        X_masked_out = (1 - M) * linear_n_octave_spectrogram_X

        # Step 4: Convert masked 3-octave band spectrograms to audio waveforms and then get new CLAP embeddings
        masked_audio_waveform_in = self.    _spectrogram_n_octave_to_clap_waveform(
            X_masked_in, audio_waveforms.shape, sampling_rate
        )
        masked_audio_waveform_out = self.    _spectrogram_n_octave_to_clap_waveform(
            X_masked_out, audio_waveforms.shape, sampling_rate
        )

        preprocessed_masked_audio_in = clap_model.preprocess_audio(masked_audio_waveform_in.tolist(), True)
        preprocessed_masked_audio_out = clap_model.preprocess_audio(masked_audio_waveform_out.tolist(), True)

        preprocessed_masked_audio_in = preprocessed_masked_audio_in.squeeze(1)
        preprocessed_masked_audio_out = preprocessed_masked_audio_out.squeeze(1)

        with torch.no_grad(): # Embeddings for loss calculation should be detached
            h_masked_in = self.clap_audio_encoder(preprocessed_masked_audio_in)[0][0]
            h_masked_out = self.clap_audio_encoder(preprocessed_masked_audio_out)[0][0]

        # Step 5: Get classifier predictions on masked inputs
        logits_masked_in = self.classifier(h_masked_in)
        logits_masked_out = self.classifier(h_masked_out)

        # Step 6: Calculate L_in and L_out
        L_in = self.cross_entropy_loss(logits_masked_in, y_hat)
        L_out = self.cross_entropy_loss(logits_masked_out, y_hat)

        # Step 7: Calculate regularization term R(M_theta(h))
        R_sparsity = torch.norm(M, p=1) / M.numel()

        R_guidance = torch.tensor(0.0).to(device)
        if self.lambda_g > 0:
            R_guidance = F.mse_loss(X_masked_in, linear_n_octave_spectrogram_X)

        R_term = self.lambda_s * R_sparsity + self.lambda_g * R_guidance

        # Step 8: Combine all terms for the total masking loss
        total_masking_loss = (self.lambda_in * L_in) - (self.lambda_out * L_out) + R_term

        return total_masking_loss, M, y_hat

    # La funzione `generate_listenable_interpretation` ora usa la nuova logica di ricostruzione
    def generate_listenable_interpretation(self, masked_n_octave_spectrogram_linear,
                                    track_name, sampling_rate, original_audio_shape):
        """
        Generates a listenable audio interpretation from the masked 3-octave band spectrogram
        using Griffin-Lim.
        """
        # Griffin-Lim è gestito dalla funzione interna     _spectrogram_n_octave_to_clap_waveform
        # Questo è per generare un file salvato, non un tensore di output diretto.

        # Per coerenza, passiamo il tensore in batch_size=1
        masked_n_octave_spectrogram_linear_batched = masked_n_octave_spectrogram_linear.unsqueeze(0)

        # Chiama la funzione interna che esegue Griffin-Lim e salva il file.
        # Nota: la funzione     _spectrogram_n_octave_to_clap_waveform restituisce un tensore,
        # ma il tuo scopo finale è salvare il file.
        # Dovrai modificare quella funzione per salvare anche il file, o farlo qui.

        reconstructed_audio_tensor = self.    _spectrogram_n_octave_to_clap_waveform(
            masked_n_octave_spectrogram_linear_batched, original_audio_shape, sampling_rate, track_name_suffix="interpreted"
        )

        # Ora che hai il tensore audio, puoi salvarlo.
        reconstructed_audio_np = reconstructed_audio_tensor.squeeze(0).detach().cpu().numpy()
        reconstructed_name = f"{track_name.replace('.pt', '')}_reconstructed.wav"

        # Crea la directory 'output' se non esiste
        output_dir = "output_reconstructed_audio"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, reconstructed_name)

        # TODO: set correct reconstructed audio path for generate_listenable_interpretations
        librosa.output.write(output_path, reconstructed_audio_np, sampling_rate)

        print(f"Generated listenable interpretation saved to: {output_path}")
        return reconstructed_audio_tensor # Restituisci il tensore per coerenza


### Explainability pipeline ###

def LMAC_explainer(tr_spectrograms, tr_audio_waveforms, tr_y_true_labels, learning_rate, opt_config):
    # TODO: finish fixing and honing the pipeline
    # TODO: correctly set up multi-GPU mode
    # TODO: finish writing comments
    
    # Inizializza il dummy CLAP model
    clap_model, audio_embedding, _ = CLAP_initializer()

    finetuned_classifier = FinetunedModel(classes=dummy_classes, device=device)

    # Decoder_output_spectrogram_shape è la forma desiderata della maschera,
    # che deve corrispondere alla forma degli spettrogrammi a 3 bande d'ottava.
    decoder_output_spectrogram_shape = tr_spectrograms.shape # (batch_size, 1, num_octave_bands, time_frames)

    decoder = Decoder(latent_dim_input=latent_dim_clap_embedding,
                      output_spectrogram_shape=decoder_output_spectrogram_shape)

    lmac_model = LMAC(classifier=finetuned_classifier,
                      decoder=decoder,
                      clap_audio_encoder=clap_model.audio_encoder,
                      alpha=1.0,
                      beta=1.0,
                      lambda_in=1.0,
                      lambda_out=1.0,
                      lambda_g=0.01,
                      lambda_s=0.001)

    optimizer, with_epochs = build_optimizer(opt_config, finetuned_classifier)

    print("\n--- Starting Decoder Training (Conceptual Loop) ---")
    for epoch in range(epochs):
        optimizer.zero_grad()

        loss, generated_mask, predicted_class_y_hat = lmac_model.calculate_masking_loss(
            linear_n_octave_spectrogram_X=tr_spectrograms,
            audio_waveforms=tr_audio_waveforms,
            y_true_labels=tr_y_true_labels,
            sampling_rate=sampling_rate # Passa il sampling_rate
        )

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    print("--- Decoder Training Finished ---")

    # --- Generazione di un'Interpretazione Audio ---
    print("\n--- Generating a Listenable Interpretation Example ---")
    # Prendi il primo campione del batch
    masked_spectrogram_for_interp = generated_mask[0, :, :, :] * tr_spectrograms[0, :, :, :]

    # Poiché `listenable_wav_from_3octaveband` salva un file e non restituisce un tensore diretto
    # la funzione `generate_listenable_interpretation` in LMAC è stata modificata per gestire il salvataggio.
    # Devi anche passare la forma dell'onda audio originale (batch_size, num_samples) per il padding.
    original_audio_shape_for_interp = tr_audio_waveforms[0].shape # Forma del singolo campione

    listenable_output_tensor = lmac_model.generate_listenable_interpretation(
        masked_spectrogram_for_interp, tr_set_tracks[0] if tr_set_tracks else "dummy_track",
        sampling_rate, original_audio_shape_for_interp
    )

    if listenable_output_tensor is not None:
        print(f"Conceptual listenable output tensor shape: {listenable_output_tensor.shape}")
