Report del task «Train EchoWorld on ultrasound data» — 3 novembre 2026
===================================================================

Il task è in pausa su richiesta dell'utente. Il training `stage` è terminato a
10.000 step; non ci sono training del task attivi. L'estrazione è sospesa con
SIGSTOP sul gruppo di processi 1886592, inclusi i quattro worker. Questo numero
identifica solo il processo di questa sessione: prima di un eventuale SIGCONT
occorre verificare che identifichi ancora il comando `mocap.prepare` atteso.
Sono presenti 93 cache di episodio complete; il dataset originale contiene 168
episodi. Il manifest completo `data/ifl/manifest.json` non è ancora stato creato.

Il risultato attuale è una pipeline funzionante che genera immagini, con qualità
ancora limitata. È un adattamento del codice EchoWorld: riusa encoder e predictor
della repository, aggiunge il percorso di condizionamento sul moto e due decoder
di immagini. I pesi sono stati addestrati da zero sui dati locali, senza un
checkpoint ufficiale degli autori. Il decoder a diffusione è un'implementazione
locale, non il decoder RCDM degli autori.

L'estrazione legge le registrazioni originali in
`/media/davide.nardi-1/CrucialX9/Nardi/ifl_recording/<nome>/<bag>` in sola lettura.
Il codice è in [mocap/bags.py](mocap/bags.py),
[mocap/prepare.py](mocap/prepare.py) e [mocap/geometry.py](mocap/geometry.py).

1. Legge immagini da `/ultrasound_image/ffmpeg`, oppure da
   `/ultrasound/image_raw` se presente. Il video è decodificato sequenzialmente,
   partendo da un keyframe e mantenendo l'associazione fra frame e timestamp.
   I pacchetti che non producono un frame utilizzabile sono registrati negli scarti.
2. Legge il tracking da `/rigid_poses` e la calibrazione da `/tf_static`.
   La posa fisica della sonda è
   `polaris_frame_T_probe = polaris_frame_T_marker @ marker_T_probe`.
   Una calibrazione mancante produce un errore.
3. Sincronizza usando i timestamp degli header. Interpola linearmente la
   traslazione e con SLERP l'orientamento: entrambe le pose ai lati devono essere
   valide e distare al massimo 50 ms dal frame. Non supera buchi di tracking,
   intervalli superiori a 100 ms o estremi senza misure di supporto.
4. Ritaglia l'immagine originale con `(left=520, top=110, right=1400, bottom=800)`,
   converte in scala di grigi e ridimensiona con padding a 224×224 mantenendo le
   proporzioni. Il crop elimina quasi tutta l'interfaccia dello scanner; rimane
   una piccola icona statica. Campiona al massimo a 5 Hz.
5. Salva per episodio `images.npy`, `poses.npz` e `metadata.json`, con timestamp,
   trasformazioni 4×4, segmenti di tracking valido, calibrazione, impostazioni e
   statistiche degli scarti. Le firme di file sorgente e impostazioni impediscono
   di mescolare cache incompatibili. Un episodio completo può essere riutilizzato.

RGB-D, forza, point cloud e traiettorie canonicalizzate non sono input del modello.
I loro byte possono comunque condividere blocchi MCAP compressi con i topic
necessari. Il disco risultava collegato a USB 2.0 (480 Mb/s); questo rallenta
l'estrazione. Durante questa sessione si è verificata un'interruzione I/O:
il primo tentativo è fallito, poi il disco è tornato disponibile e l'esportazione
è ripartita. I log sono in `runs/full_export_20261103/`, incluso
`first_attempt_errors.json`; il secondo tentativo non aveva errori registrati
al momento della pausa.

La cache si trova fisicamente in `finetune/data/`. Ho ripristinato il collegamento
`data -> finetune/data`, perché configurazioni e checkpoint si riferivano al
percorso `data/`. Il manifest usato per il resume è identico, anche nella firma
SHA-256, a quello del checkpoint. Cache e risultati sono esclusi da Git.

Il dataset di training viene costruito da [mocap/dataset.py](mocap/dataset.py).
Ogni esempio contiene l'ecografia iniziale `I_a`, l'ecografia bersaglio `I_b` e
il movimento relativo `DeltaT = inverse(T_a) @ T_b`. Il movimento diventa un
vettore di sei numeri: tre traslazioni in millimetri e un vettore di rotazione
asse-angolo in radianti. Non sono angoli di Eulero. Prima del predictor questi
valori vengono divisi per `[30, 30, 30, 0.3, 0.3, 0.3]`.

Le coppie distano 0,4–3 secondi, appartengono allo stesso episodio e allo stesso
segmento di tracking valido. Nel training l'ordine può essere invertito. Sono
escluse le immagini con intensità media sotto 8/255: è un filtro per immagini
quasi nere, non un'annotazione di contatto. I gruppi sono separati prima di
costruire le coppie: `yuan` è validation, `xuesong` è test, gli altri gruppi sono
training. I nomi delle cartelle non sono identità di paziente verificate.

La struttura del modello è implementata in [mocap/model.py](mocap/model.py).

| Componente | Input | Output e ruolo |
|---|---|---|
| Encoder ViT | Ecografia iniziale, replicata su tre canali | Token di patch 16×16. Nel modello ViT-S, un'immagine 224×224 contiene 196 patch e ogni token ha 384 componenti. I tre canali contengono la stessa ecografia, non la camera RGB. |
| Predictor spaziale | Token visibili e posizioni delle patch nascoste | Rappresentazioni delle patch nascoste; serve a imparare struttura ecografica. |
| MLP del moto e predictor | Token dell'immagine iniziale e movimento relativo a sei componenti | Un vettore latente che rappresenta l'immagine attesa alla posa bersaglio. |
| Encoder target e proiettore target | Immagine iniziale completa o immagine bersaglio, durante il training | Rappresentazioni di supervisione; i pesi seguono una media mobile dei pesi online e non ricevono gradienti. |
| Proiettore contrastivo | Latente predetto e rappresentazione target | Vettori da 128 componenti per distinguere il bersaglio corretto dagli altri esempi del batch. |
| Decoder convoluzionale | Vettore latente predetto | Ecografia in scala di grigi. Non riceve direttamente l'immagine bersaglio o un collegamento di copia dell'immagine iniziale. |

In inferenza il percorso è `immagine iniziale + movimento relativo -> encoder e
predictor -> latente predetto -> decoder -> ecografia generata`. La traiettoria
da sola non specifica l'anatomia: serve un'immagine iniziale del soggetto.

Il primo training aggiorna congiuntamente encoder, predictor, proiettore e
decoder convoluzionale. Nasconde casualmente il 50% delle patch del contesto e
combina cinque obiettivi:

`L = L_spatial + 0.1 L_motion + 0.1 L_latent + 5 L_pixel + L_reconstruction`.

`L_spatial` ricostruisce le rappresentazioni nascoste; `L_motion` è contrastiva
fra predizione e bersaglio; `L_latent` allinea direttamente i due latenti;
`L_pixel` confronta la predizione con l'ecografia reale, dando più peso alle zone
non nere e includendo scale più grossolane; `L_reconstruction` insegna al decoder
a ricostruire anche dal latente target. Le ultime estensioni e varie scelte di
architettura/training sono adattamenti locali, non una riproduzione integrale
del protocollo del paper.

Il secondo training, opzionale, usa [mocap/diffusion.py](mocap/diffusion.py).
Congela il world model addestrato e aggiorna una U-Net condizionata sul suo
latente predetto. Durante il training aggiunge rumore all'ecografia bersaglio e
impara a rimuoverlo; l'esperimento completato più riuscito usa la parametrizzazione
`v`. Durante la generazione parte da rumore, senza leggere il bersaglio, e usa
40 passi DDIM, guidance 1,5 e seed 42. Le condizioni sono applicate ai blocchi
della U-Net tramite scale e offset appresi (FiLM). Questo decoder aggiunge texture,
ma la texture plausibile non prova che la struttura anatomica sia corretta.

Il training usa GPU 0, una RTX PRO 6000 Blackwell da circa 96 GB, PyTorch, BF16,
AdamW, warmup e learning rate coseno. Per ViT-S il batch è 48 e il learning rate
iniziale configurato è 0,0003; per la diffusione pilota il batch è 32 e il learning
rate è 0,0002. Si controllano loss e gradienti finiti e si limita la norma dei
gradienti. GPU 1 ospita un altro processo e non è stata usata per questo task.
Uno step è un aggiornamento su un batch: gli step riutilizzano immagini e coppie,
non corrispondono ad altrettanti nuovi frame registrati.

| Run | Dati e modello | Stato |
|---|---|---|
| `runs/pilot/` | 18 episodi totali; modello 128×128, encoder 192×6 blocchi, predictor 4 blocchi | 2.000 step completati; arresto e ripresa reali verificati. |
| `runs/small_pilot/` | Stessi 18 episodi; ViT-S 224×224, 384 componenti, encoder 12 blocchi, predictor 6 blocchi | 6.000 step completati; miglior checkpoint allo step 5.000. |
| `runs/diffusion_pilot/` | Primo decoder a diffusione con predizione epsilon | Prova fermata a 2.000 step per risultati scadenti. |
| `runs/diffusion_v_pilot/` | Decoder a diffusione sul world model pilota congelato | 10.000 step completati. |
| `runs/stage/` | Manifest fisso di 70 episodi: 66 train, 2 validation, 2 test; inizializzato dal miglior `small_pilot` | Ripreso dal checkpoint 1.000 e completato a 10.000; miglior checkpoint allo step 1.500. |

Il manifest `stage` contiene 3.685 frame train, 198 validation e 114 test, prima
del filtro delle immagini quasi nere. Il loader train ha 2.940 indici sorgente
utilizzabili, ciascuno con scelta del frame bersaglio. Il training sul dataset
completo di 168 episodi non è ancora stato eseguito. Non è iniziato un nuovo
training di diffusione sul modello `stage`.

Per ogni run `best.pt` è scelto dal MAE sulla validation e `last.pt` permette
la ripresa. I checkpoint includono modello, EMA, ottimizzatore, step,
configurazione, stato casuale e firma del manifest. Il resume rifiuta configurazioni
o manifest diversi. `--init-from`, usato per passare da `small_pilot` a `stage`,
carica invece i pesi e ricomincia ottimizzatore e contatore.

I file principali sono:

- `runs/<run>/config.json`: configurazione effettiva.
- `runs/<run>/train.jsonl`: loss e avanzamento.
- `runs/<run>/status.json`: step finale e motivo di arresto, quando terminato.
- `runs/<run>/val_<step>/`: metriche e immagini di validation durante il training.
- `runs/<run>/evaluation*/`: valutazioni separate e riproducibili.
- `runs/<run>/trajectory/`: PNG, `generated.npy`, GIF, confronto e metadati.

Le valutazioni confrontano la predizione con il bersaglio, con la copia del
contesto e con la predizione ottenuta azzerando il moto. Riportano MAE, PSNR,
SSIM locale e MAE nelle zone non nere. Ho aggiunto `per_pair_metrics.csv` per
tracciare episodio, indici, movimento ed errore di ogni esempio; i nuovi
`metrics.json` riportano anche checkpoint, step, risoluzione e firma del manifest.
Queste valutazioni riguardano la validation; il test riservato non è stato usato
per scegliere checkpoint.

Per il miglior checkpoint `stage` (step 1.500), le ultime misure sono:

| Valutazione | Coppie | MAE modello | MAE copia contesto | MAE moto zero | SSIM modello / copia |
|---|---:|---:|---:|---:|---:|
| Circa 1 secondo | 118 | 0,03755 | 0,02901 | 0,03825 | 0,6718 / 0,7474 |
| Circa 3 secondi | 106 | 0,04172 | 0,04063 | 0,04186 | 0,6520 / 0,6806 |

Le intensità sono in [0,1]; MAE più basso è meglio. La prima misura è su fino a
64 coppie per episodio, mentre la selezione durante il training usava fino a 32:
per questo il MAE 0,03755 differisce dal valore 0,03783 usato per selezionare
lo stesso checkpoint. Le metriche di run a 128×128 e 224×224 non vanno confrontate
come se il protocollo fosse identico. Il modello non supera la copia del contesto
nel MAE complessivo e nello SSIM di queste prove, anche se migliora alcune metriche
nelle regioni non nere. Non è ancora dimostrata una simulazione anatomica fedele.

I risultati da aprire sono:

- [Confronto stage migliore](runs/stage/evaluation/comparison.png): colonne
  contesto, predizione, bersaglio, moto zero, errore assoluto moltiplicato per tre.
- [Metriche stage](runs/stage/evaluation/metrics.json) e
  [CSV per coppia](runs/stage/evaluation/per_pair_metrics.csv).
- [Checkpoint stage migliore](runs/stage/best.pt).
- [Sequenza stage](runs/stage/trajectory/sequence.png) e
  [GIF stage](runs/stage/trajectory/generated.gif).
- [Sequenza diffusione pilota](runs/diffusion_v_pilot/trajectory/sequence.png) e
  [GIF diffusione pilota](runs/diffusion_v_pilot/trajectory/generated.gif).

Nei confronti di sequenza la riga superiore è generata e quella inferiore è reale.
Il modello è addestrato su coppie. Per una traiettoria, la modalità predefinita
genera separatamente ogni frame dalla stessa immagine iniziale e dal movimento
relativo fra la posa iniziale e quella richiesta. Non c'è uno stato ricorrente
appreso lungo tutta la sequenza. `--rollout` reinserisce invece il frame generato
come contesto successivo e può accumulare errore. Le immagini future del dataset
servono solo al confronto visivo, non all'inferenza. Una traiettoria esterna usa
un PNG iniziale e un NPZ con `world_T_probe` di forma `(N+1,4,4)`, traslazioni in
metri, nel riferimento della sonda calibrata.

La suite aggiornata è passata: **15 test**, inclusi decodifica H.264, geometria,
sincronizzazione, separazione dei gruppi, gradienti, checkpoint, generazione,
uguaglianza del resume su CPU e tracciabilità del CSV. I comandi riproducibili e
le convenzioni complete sono in [README_MOCAP.md](README_MOCAP.md).

Il lavoro restante è completare l'esportazione, addestrare e validare sul manifest
completo, quindi valutare il modello selezionato sul test riservato. L'eventuale
ulteriore lavoro sulla diffusione deve essere misurato per fedeltà al bersaglio,
oltre che per qualità visiva. Queste attività restano in pausa.
