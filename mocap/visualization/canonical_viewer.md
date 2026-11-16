# Viewer canonico per soggetto e medico

La pagina servita da `http://localhost:8070/canonical` consente di scegliere un
singolo soggetto e confrontare i medici dalla legenda. I colori sono stabili:
Amir viola, Davide blu, Feng verde, Yuan arancione. Il numero accanto al medico
indica le sue registrazioni per il soggetto selezionato. Assi sonda, inizio e fine
seguono gli stessi filtri delle curve.

La mappa è in `annotations/recording_participants.json`:

| Prefisso / numeri | Medico | Soggetto |
| --- | --- | --- |
| expert1–18 | Amir | Yuan |
| expert19–38 | Amir | Davide |
| yuan (anche suffissi come yuan9b) | Davide | Yuan |
| davide | Yuan | Davide |
| nardi | Feng | Davide |
| altri | Davide | prefisso del bag senza numero |

`nardi` e `davide` vengono accorpati nel soggetto Davide, conservando i medici
distinti. La mappa serve solo alla visualizzazione: non riscrive i gruppi del
manifest, gli split o le pose del training.

Rigenerare entrambe le viste dalla directory `EchoWorld`:

```bash
.venv/bin/python -m mocap.visualization.canonical_dataset \
  --manifest data/ifl_canonical/manifest.json \
  --body ../Canonical_US/outputs/probe_reconstruction_trims/shard0/canonical_smpl.ply
.venv/bin/python -m mocap.visualization.canonical_dataset \
  --manifest data/ifl_canonical/manifest.json \
  --body ../Canonical_US/outputs/probe_reconstruction_trims/shard0/canonical_smpl.ply \
  --zoom
```

Il server legge direttamente gli HTML aggiornati, senza riavvio. Il secondo è
servito da `/canonical/zoom`. `?subject=Yuan` preseleziona il soggetto. Gli HTML
sono autonomi, con Plotly incorporato, e funzionano anche senza server.

Le curve mantengono il sollevamento di visualizzazione originale di 4 mm lungo
`-Z` sonda (`--lift-mm 0` lo disabilita); le pose salvate non vengono modificate.
Il campionamento preserva gli estremi reali e le interruzioni di segmento.
