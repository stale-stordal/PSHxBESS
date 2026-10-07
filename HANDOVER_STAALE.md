Hei Ståle,

Lykke til med det videre arbeidet med artikkelen!

## Om repoet

Repoet inneholder forhåpentligvis alle filene som trengs for å kjøre analysene på nytt. Det opprinnelige masterrepositoryet ble imidlertid ganske uoversiktlig mot slutten av arbeidet, da flere filer ble for store til å kunne pushes med Git, og det måtte mye manuell flytting av filer til og fra Solstorm, inkludert filer som bare eksisterte i Solstrom grunnet størrelsen.

Det finnes også en mer detaljert, AI-generert kjørebeskrivelse i [`RUN_PIPELINE.md`](RUN_PIPELINE.md). Gi gjerne beskjed dersom du oppdager at filer mangler eller ser ut til å være utdaterte, så skal vi forsøke å finne fram det som eventuelt mangler.

## Kjøre-rekkefølge

Analysen består av tre hovedsteg:

1. Stokastisk modellering av spot- og FCR-priser
2. Optimering av batteri- og pumpekraftverk
3. Real options analysis (ROA)

### 1. Stokastisk modellering

Først kalibreres og simuleres de stokastiske pris- og FCR-modellene. Modellene produserer simulerte prisbaner som brukes som input til optimeringen. Koden ligger i mappen `stochastic_modelling/`.

Kalibreringen kjøres med `calibrate_joint.py`, og simuleringen av prisbanene med `forecast_joint.py`. Se [`RUN_PIPELINE.md`](RUN_PIPELINE.md) for detaljer om inputfiler, konfigurasjon og rekkefølgen på kommandoene.

Prisbanene blir lagret som `.parquet`-filer. De produserte filene er svært store, normalt over 10 GB totalt, og ble derfor ikke hentet ned fra Solstorm. Dersom du ser referanser til `.csv`-filer i enkelte skript eller kommentarer, skyldes det at vi først brukte CSV-format før vi gikk over til Parquet for å redusere lagringsbehov og kjøretid.

### 2. Optimering

Optimeringen bruker de simulerte prisbanene fra steg 1 og beregner dispatch og inntekter for BESS og PSH.

Denne delen ble kjørt på Solstorm med omfattende parallellisering. Den nåværende konfigurasjonen er derfor i praksis lite egnet for en vanlig laptop, særlig med mange prisbaner. Vi anbefaler å bruke Solstorm dersom du ønsker å kjøre optimeringen på nytt.

Optimeringen krever også en fungerende Gurobi-installasjon og en gyldig lisens. Resultatene lagres i `optimization/results/paths/`.

### 3. Real options analysis (ROA)

ROA-koden ble kjørt på lokale maskiner. Den bruker resultatene fra optimeringen som input, nærmere bestemt de kombinerte resultatfilene:

- `roa/input_files/BESS_optimized_results_combined.csv`
- `roa/input_files/PSH_optimized_results_combined.csv`

Dersom du bare ønsker å kontrollere ROA-beregningene, kan du derfor hoppe over de to første stegene og kjøre ROA direkte med disse filene. Se [`RUN_PIPELINE.md`](RUN_PIPELINE.md) for kommandoer og mer informasjon om dette.




Mathea, Oskar og Einars

