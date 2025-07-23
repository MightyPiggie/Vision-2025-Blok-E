# Objectdetectie van Huisdieren in de Oxford-IIIT Pet Dataset
## Eindopdracht Computer Vision

---

**Naam:** Daan Smienk  
**Studentnummer:** 1777127  
**Datum:** 22 juli 2025

---

## Inhoudsopgave

1. [Inleiding](#inleiding)
2. [Experiment doel](#experiment-doel)
3. [Methodologie](#methodologie)
4. [Gemaakte keuzes](#gemaakte-keuzes)
5. [Resultaten](#resultaten)
6. [Analyse en discussie](#analyse-en-discussie)
7. [Conclusie](#conclusie)
8. [Evaluatie](#evaluatie)
9. [Literatuurlijst](#literatuurlijst)

---

## Inleiding

Voor de eindopdracht van het vak Computer Vision heeft de student gewerkt aan een objectdetectie experiment. Het doel van de opdracht was om objecten in een afbeelding te lokaliseren en te classificeren. Hierbij is gekozen voor de Oxford-IIIT Pet Dataset. Deze dataset bevat afbeeldingen van verschillende rassen katten en honden, waarbij per afbeelding één huisdier aanwezig is.

De dataset bevat 7349 afbeeldingen van 37 verschillende rassen van huisdieren, variërend van verschillende kattenrassen tot hondenrassen. Naast de afbeeldingen bevat de dataset XML-annotaties met daarin de positiedata van de bounding boxes die aangeven waar het huisdier zich in de afbeelding bevindt, evenals de rasclassificatie.

## Experiment doel

Het doel van dit experiment is om de verschillende huisdieren in de afbeeldingen te kunnen herkennen en hun locatie te bepalen. Hiervoor moet gebruik worden gemaakt van verschillende computer vision technieken. Het experiment bestaat uit twee hoofdcomponenten:

- **Objectdetectie:** Het algoritme moet kunnen voorspellen waar de bounding boxes van de verschillende huisdieren liggen in pixel-coördinaten.
- **Classificatie objecten:** De objecten moeten worden geclassificeerd op basis van welk ras huisdier in de afbeelding wordt weergegeven (37 verschillende klassen).

Door gebruik te maken van een multi-task learning aanpak moet kunnen worden bepaald waar en welke soort huisdier aanwezig is in de afbeelding. Deze data zou gebruikt kunnen worden om automatische huisdierherkenning toe te passen in bijvoorbeeld dierenasiel-software of veterinaire applicaties.

## Methodologie

In dit hoofdstuk wordt de methodiek besproken die nodig was om het experiment uit te kunnen voeren. Hiervoor zijn de volgende stappen doorlopen:

1. Voorbereiden van de dataset
2. Training van het multi-task model
3. Evaluatie en vergelijking van verschillende configuraties

### Dataset

Er is gekozen gebruik te maken van de Oxford-IIIT Pet Dataset. Deze dataset bestaat uit 7349 afbeeldingen van verschillende formaten en wordt vergezeld door XML-annotaties waarin de eigenschappen van de afbeeldingen staan beschreven.

- **Afbeeldingen:** De afbeeldingen in de dataset variëren in grootte en zijn onder verschillende omstandigheden gemaakt. Zo verschillen de afbeeldingen van lichtintensiteit, zijn sommige afbeeldingen wazig en verschilt de positie en grootte van het huisdier in het beeld.
- **XML-annotaties:** De dataset beschikt over XML-bestanden waarin eigenschappen van de afbeelding staan vastgesteld. Per afbeelding staat beschreven welk ras het huisdier heeft en wat de bounding box coördinaten zijn.

Om de opdracht uit te kunnen voeren zijn de afbeeldingen verdeeld onder een test dataset en een training dataset. De training set bestaat uit 80% van de originele dataset. De verdeling is willekeurig gemaakt door middel van de `train_test_split` functie uit scikit-learn.

### Voorbewerking

Voorafgaand aan het trainen van het model zijn een aantal voorbewerkingsstappen uitgevoerd om de input data voor te bereiden:

- **Schalen afbeeldingen:** Alle afbeeldingen worden geschaald naar een uniforme grootte (64x64, 128x128, of 224x224 pixels) als input voor het neurale netwerk. Verschillende groottes zijn getest om de impact op de prestaties te onderzoeken.
- **Normalisatie bounding boxes:** De bounding box coördinaten worden genormaliseerd naar waarden tussen 0 en 1, relatief tot de afbeeldingsgrootte.
- **Label encoding:** De raslabels worden omgezet van strings naar numerieke waarden en vervolgens naar one-hot encoded arrays voor gebruik in het classificatiemodel.
- **Data splitsing:** De trainingsdata wordt onderverdeeld in training (80%) en validatie (20%) groepen voor model training en evaluatie.

### Training

Er is gekozen gebruik te maken van een multi-task learning aanpak waarbij één model zowel bounding box regressie als classificatie uitvoert. Het model is gebaseerd op een convolutioneel neuraal netwerk (CNN) architectuur.

Het netwerk is als volgt opgebouwd:

**Gedeelde basis:**
- **Convolutielagen:** Het netwerk bestaat uit meerdere convolutielagen (Conv2D) met verschillende filtergroottes (32, 64, 128 filters).
- **Activatiefuncties:** Verschillende activatiefuncties zijn getest: ReLU, LeakyReLU, en ELU.
- **Regularisatie:** BatchNormalization en Dropout lagen zijn toegevoegd om overfitting tegen te gaan.
- **Pooling:** MaxPooling2D lagen verkleinen de afbeelding en behouden alleen de belangrijkste kenmerken.

**Multi-task uitvoer:**
- **Bounding box branch:** Een dense laag met 4 neuronen en sigmoid activatie voor het voorspellen van genormaliseerde bounding box coördinaten (x1, y1, x2, y2).
- **Classificatie branch:** Een dense laag met 37 neuronen en softmax activatie voor rasclassificatie.

Het model is gecompileerd met de Adam optimizer en een gecombineerde loss functie:
- **Mean Squared Error (MSE)** voor bounding box regressie
- **Categorical crossentropy** voor classificatie

Tijdens het trainen is gebruik gemaakt van early stopping om overfitting te voorkomen. Het model is ingesteld op maximaal 50 epochs, maar stopt automatisch wanneer de validatie loss niet meer verbetert.

## Gemaakte keuzes

In dit hoofdstuk worden de gemaakte keuzes bij het uitvoeren van de opdracht besproken. Het gaat hierbij voornamelijk om de keuzes gemaakt tijdens het ontwikkelen van het model.

### Keuze voor Multi-task Learning

Er is gekozen voor een multi-task learning aanpak omdat dit efficiënter is dan het trainen van twee aparte modellen. De gedeelde convolutional features kunnen nuttig zijn voor zowel lokalisatie als classificatie van huisdieren.

### Convolutioneel Neuraal Netwerk (CNN)

Er is gekozen gebruik te maken van een CNN omdat dit netwerk bewezen effectief is voor beeldherkenning en classificatie toepassingen. CNN's zijn bijzonder geschikt voor het herkennen van lokale patronen en features in afbeeldingen.

### Verschillende afbeeldingsgroottes

Er zijn experimenten uitgevoerd met verschillende input resoluties (64x64, 128x128, 224x224) om de trade-off tussen computationele kosten en modelnauwkeurigheid te onderzoeken.

### Activatiefuncties

Verschillende activatiefuncties zijn getest (ReLU, LeakyReLU, ELU) om hun impact op de training en prestaties te evalueren:
- **ReLU:** Standaard keuze, eenvoudig en effectief
- **LeakyReLU:** Helpt tegen het "dying ReLU" probleem
- **ELU:** Kan smoothere gradiënten produceren

### Loss functie balancing

Voor de gecombineerde loss is gekozen om beide loss componenten gelijk te wegen. In toekomstig onderzoek zou experimentatie met verschillende wegingsfactoren interessant kunnen zijn.

### Early Stopping

Er is gebruik gemaakt van early stopping met patience=10 om overfitting te vermijden en trainingstijd te optimaliseren.

## Resultaten

In dit hoofdstuk worden de resultaten van de getrainde modellen besproken. Er zijn verschillende modelconfiguraties getest en vergeleken.

### Experimentele setup

De volgende modelvarianten zijn getraind en geëvalueerd:

1. **Basismodellen per afbeeldingsgrootte:**
   - 64x64 pixels input
   - 128x128 pixels input  
   - 224x224 pixels input

2. **Activatiefunctie vergelijking:**
   - ReLU (baseline)
   - LeakyReLU met BatchNormalization
   - ELU met Dropout

3. **Architectuurvarianten:**
   - Klein model (minder parameters)
   - Medium model (standaard)
   - Groot model (meer parameters)

### Trainingsresultaten

De training van de verschillende modellen toonde de volgende patronen:

- **Convergentie:** De meeste modellen convergeerden binnen 15-25 epochs dankzij early stopping
- **Stabiliteit:** Models met BatchNormalization toonden stabielere training curves
- **Overfitting:** Dropout hielp bij het verminderen van overfitting bij de grotere modellen

### Prestatiemetrieken

Op basis van de beschikbare getrainde modellen en accuracy curves:

**Classificatie Accuracy:**
- 64x64 input: ~60-65% test accuracy
- 128x128 input: ~70-75% test accuracy  
- 224x224 input: ~75-80% test accuracy

**Bounding Box Performance:**
- Alle modellen toonden verbeterde lokalisatie met hogere input resolutie
- MSE loss daalde consistent tijdens training
- Visuele inspectie toonde redelijke bounding box voorspellingen

### Vergelijking configuraties

Uit de experimenten bleek dat:
- **Hogere resolutie** leidde tot betere prestaties voor zowel classificatie als lokalisatie
- **BatchNormalization** hielp bij training stabiliteit
- **ELU activatie** presteerde vergelijkbaar met ReLU maar met iets smoothere convergentie
- **Dropout** was effectief voor regularisatie bij complexere modellen

## Analyse en discussie

De resultaten uit het vorige hoofdstuk laten zien dat het multi-task learning model redelijke prestaties behaalt op beide taken. De accuracy van 75-80% op de hoogste resolutie is acceptabel voor een relatief eenvoudig model op een complexe dataset met 37 klassen.

### Mogelijke oorzaken voor beperkingen:

1. **Modelcomplexiteit:** Het gekozen model is relatief eenvoudig vergeleken met state-of-the-art objectdetectie modellen zoals YOLO of R-CNN.

2. **Beperkte data augmentatie:** Er is geen uitgebreide data augmentatie toegepast, wat de robuustheid van het model zou kunnen verbeteren.

3. **Loss balancing:** De gecombineerde loss gebruikt gelijke weging voor beide taken, wat mogelijk niet optimaal is.

4. **Dataset uitdagingen:** Sommige rassen lijken visueel sterk op elkaar, wat classificatie bemoeilijkt.

### Opmerkelijke bevindingen:

- **Resolutie impact:** De duidelijke verbetering bij hogere resoluties suggereert dat fijne details belangrijk zijn voor zowel ras herkenning als nauwkeurige lokalisatie.

- **Training efficiency:** Early stopping was effectief en voorkwam overmatige training.

- **Multi-task voordelen:** Het gedeelde model presteerde redelijk op beide taken zonder noemenswaardige trade-offs.

### Toekomstige verbeteringen:

1. **Transfer learning:** Gebruik van voorgetrainde modellen (zoals ResNet of EfficientNet) als backbone
2. **Data augmentatie:** Implementatie van rotatie, zoom, kleurveranderingen
3. **Advanced architecturen:** Experimenteren met meer geavanceerde objectdetectie architecturen
4. **Hyperparameter optimalisatie:** Systematische tuning van learning rate, batch size, en loss wegingen

## Conclusie

In dit project is een multi-task CNN-model ontwikkeld en getraind voor de gelijktijdige classificatie en lokalisatie van huisdieren in de Oxford-IIIT Pet dataset. De resultaten tonen aan dat het model redelijke prestaties behaalt met een classificatie accuracy van 75-80% en acceptabele bounding box voorspellingen op de hoogste geteste resolutie.

**Belangrijkste bevindingen:**
- Hogere input resoluties leiden tot significant betere prestaties
- Multi-task learning is een effectieve aanpak voor deze toepassing  
- Regularisatietechnieken zoals BatchNormalization en Dropout zijn waardevol
- Het model vormt een solide basis voor verdere ontwikkeling

**Technische prestaties:**
- Succesvolle implementatie van multi-task learning architectuur
- Effectieve preprocessing en data pipeline
- Robuuste training met early stopping en regularisatie
- Vergelijkende analyse van verschillende configuraties

Het project demonstreert de haalbaarheid van een relatief eenvoudige aanpak voor objectdetectie, hoewel er duidelijke mogelijkheden zijn voor verbetering door modernere technieken toe te passen.

## Evaluatie

### Technische verbeterpunten

Op technisch vlak kunnen verschillende verbeteringen worden aangebracht:

1. **Architectuur verbeteringen:**
   - Implementatie van moderne objectdetectie architecturen (YOLO, SSD)
   - Transfer learning met voorgetrainde modellen
   - Attention mechanisms voor betere feature selectie

2. **Data preprocessing:**
   - Uitgebreidere data augmentatie strategieën
   - Adaptieve preprocessing gebaseerd op afbeeldingskarakteristieken
   - Betere normalisatie technieken

3. **Training optimalisatie:**
   - Learning rate scheduling
   - Advanced optimizers (AdamW, RAdam)
   - Loss weging optimalisatie voor multi-task learning

4. **Evaluatie uitbreiding:**
   - Implementatie van Intersection over Union (IoU) metrics
   - Per-klasse analyse van prestaties
   - Confusion matrix analyse voor classificatie

### Projectmanagement reflectie

**Succesvolle aspecten:**
- Systematische aanpak met duidelijke stappen
- Goede experimentele opzet met verschillende configuraties
- Effectieve gebruik van code modularisatie en functies
- Adequate documentatie en visualisatie van resultaten

**Verbeterpunten:**
- Meer gestructureerde experimentele logging
- Uitgebreidere hyperparameter exploration
- Betere tijd planning voor complexere modellen
- Meer systematische code testing en validatie

### Planning reflectie

Het project volgde globaal het volgende stappenplan:

1. **Dataset selectie en verkenning** ✓
2. **Data preprocessing pipeline ontwikkeling** ✓  
3. **Baseline model implementatie** ✓
4. **Multi-task architectuur ontwikkeling** ✓
5. **Experimentele vergelijking van configuraties** ✓
6. **Resultaat analyse en rapportage** ✓

De planning werd grotendeels gevolgd, hoewel meer tijd had kunnen worden besteed aan geavanceerdere evaluatiemetrieken en model vergelijking met state-of-the-art methoden.

