# L'algorithme KNN
Dans le cadre de ce projet, des documents texte de recettes sont traités. Ces documents sont stockés dans les dossiers "plats", "desserts" et "entrees", contenant chacun cinq fichiers de recettes. Ces dossiers se trouve dans un autre dossier nommé "KNN_corpus". Les fichiers passent par les méthodes de la classe `TextVect` pour être lus, tokenizés et vectorisés. Les vecteurs obtenus sont utilisés pour calculer le TF-IDF de chaque terme dans chaque document, permettant ainsi d'obtenir des vecteurs normalisés de poids de termes pour chaque document. Les résultats du TF-IDF sont stockés dans la variable `data` de la classe `KNNClass`. Ensuite, lorsqu'un nouveau texte à classifier est fourni, il est également converti en un vecteur TF-IDF et comparé aux vecteurs stockés dans `data` à l'aide de la méthode `classify` qui se trouve dans la classe `KNNClass`, en utilisant une fonction de similarité qui peut être sélectionnée parmi les trois options disponibles dans la classe `Similarity`.
En outre, la classe `Gestion` contient deux méthodes facultatives pour manipuler les données stockées dans la classe `KNNClass`. La méthode `add_class_input` permet d'ajouter des données à la classe `KNNClass` en appelant la méthode `add_class` de la classe `KNNClass`, tandis que la méthode `del_class_input` permet de supprimer des données de la classe en appelant la méthode `del_class` de la classe `KNNClass`.
Les tests de la méthode `classify` sont faits en utilisant les fichiers textes "riz_cantonais.txt" et "tiramisu_framboise.txt". 
Les traces d'exécutions de l'algorithme KNN sont mises dans le fichier "resultat_main.txt".


### Classe `KNNClass`
La classe `KNNClass` implémente un modèle de classification basé sur l'algorithme KNN (les k plus proches voisins). Cela permet de classer des vecteurs dans différentes classes en fonction de leur similarité avec les vecteurs de chaque classe. Les méthodes de cette classe sont :

* `init(self, description: str="", data=[]):` 
Cette méthode est le constructeur de la classe. Elle prend en entrée une chaîne de caractères qui décrit l'ensemble des classes et une liste de vecteurs. Elle initialise deux variables d'instance : la description de l'ensemble de classes et les données, représentées par une liste de dictionnaires où chaque dictionnaire contient un label de classe et les vecteurs associés à cette classe.

* `add_class(self, label: str, vectors: list):`
Cette méthode permet d'ajouter une nouvelle classe au modèle. Elle prend en entrée le nom de la  classe à ajouter et une liste de vecteurs à associer à cette classe. Elle vérifie d'abord si la classe n'existe pas déjà dans le modèle, sinon elle l'ajoute.

* `get_classes(self):` 
Cette méthode permet de récupérer la liste des classes actuelles.

* `add_vector(self, label: str, vector):` 
Cette méthode permet d'ajouter un vecteur à une classe existante. Elle prend en entrée le nom de la classe et le vecteur à ajouter. Elle vérifie si la classe existe dans le modèle, puis ajoute le vecteur à cette classe.

* `del_class(self, label:str):` 
Cette méthode permet de supprimer une classe existante. Elle prend en entrée le nom de la classe à supprimer et vérifie si elle existe dans le modèle. Si c'est le cas, elle la supprime.

* `save_as_json(self, filename:str):` 
Cette méthode permet de sauvegarder les données du modèle sous forme de fichier JSON. Elle prend en entrée le nom du fichier de sortie et utilise la méthode `json.dump()` pour écrire les données dans ce fichier.

* `load_as_json(self, filename:str):` 
Cette méthode permet de charger les données d'un fichier JSON dans un modèle. Elle prend en entrée le nom du fichier à charger et utilise la méthode `json.load()` pour récupérer les données du fichier.

* `classify(self, vector: dict, k: int, sim_func=None) -> List[Tuple[str, float]]:` 
Cette méthode permet de classer un vecteur avec les vecteurs des classes existantes dans le modèle. Elle prend en entrée un vecteur sous forme de hashage, le nombre de voisins les plus proches (k) à considérer et une fonction de similarité (définie dans la classe `Similarity`), (par défaut, la fonction de calcul de cosinus). Elle calcule d'abord les similarités entre le nouveau vecteur et les vecteurs de chaque classe, puis trie les résultats en fonction de la similarité décroissante. Elle renvoie ensuite une liste triée de paires `[label:str,sim:float]` pour les classes candidates. La similarité est la moyenne des similarités obtenues sur les vecteurs retenus pour la classe correspondante dans les k plus proches voisins.

**Création d'un modèle KNN**
Instanciation de la classe en fournissant une description :
```python
from knnclass import KNNClass
test_knn = KNNClass(description="données de test")
```


### Classe `TextVect`
La classe `TextVect` est une classe qui permet de représenter des fichiers texte sous forme de vecteurs de tokens. Elle fournit plusieurs méthodes qui permettent de traiter et de transformer les textes pour les rendre compatibles avec la classe `KNNClass`. Ces méthodes incluent la tokenisation des textes, la création de vecteurs de fréquence pour chaque fichier, ainsi que le filtrage des stopwords et des hapax. De plus, la classe offre une méthode de calcul du score TF-IDF pour chaque mot dans chaque vecteur, ce qui permet de pondérer les termes en fonction de leur importance relative pour la classification de la méthode `classify`. 

* `tokenize(text:str, tok_grm)->list:`
Cette fonction prend en entrée un texte à tokeniser et une expression régulière compilée pour la tokenisation. Elle renvoie la liste des tokens. Cette fonction effectue une normalisation des apostrophes et des caractères spéciaux pour un traitement correct.

* `vectorise(tokens:list)->dict:`
Cette fonction prend en entrée une liste de tokens et renvoie un dictionnaire contenant les mots (clés) et les fréquences associées (valeurs).

* `read_texts(folder_names: list) -> list :`
Cette fonction permet de lire tous les fichiers texte présents dans les dossiers de la liste folder_names, de les découper en tokens et de créer des vecteurs pour chaque fichier. Elle retourne une liste de dictionnaires chacun contenant le label du dossier et une liste de vecteurs.

* `read_txt(file_name:str) -> list :`
Cette fonction permet de lire un fichier texte, de le découper en mots ou termes et de créer un vecteur de représentation pour le fichier. Elle retourne une liste de dictionnaires contenant le label et le vecteur du fichier.

  - Notez que `read_texts` est utilisé pour la création des classes avec les vecteurs associés et que `read_txt` est utilisé pour le fichier à classifier en utilisant la méthode `classify`.
  - L'expresion regulière utilisée pour la tokenisation permet de capturer les abréviations courantes, les mots composés avec un tiret, les mots avec des apostrophes, et les mots ne contenant que des lettres. Elle ignore les chiffres car les données sont des recettes (donc ignore les mesures d'ingrédients).

* `read_dict(stoplist_filename: str) -> set(str):`
Cette fonction permet de lire une stoplist à partir d'un fichier. Le fichier doit contenir un mot par ligne. La fonction prend en entrée le nom du fichier contenant les stopwords, et renvoie un ensemble (set) des stopwords.

* `filtrage(stoplist: set, documents: list, non_hapax: bool) -> list:`
Cette fonction permet de filtrer les vecteurs des documents en éliminant les stopwords. Elle prend en entrée un ensemble de stopwords, une liste de documents, et un booléen indiquant si on veut éliminer les hapax (True) ou non (False). Elle renvoie une liste de documents filtrés. Le filtrage est effectué en parcourant chaque document, chaque vecteur de mots et en éliminant les mots appartenant à la stoplist et les hapax si non_hapax est True.

* `tf_idf (documents:list)->list:`
La fonction `tf_idf` calcule le score TF-IDF pour chaque mot dans chaque vecteur de chaque document d'une liste de documents. Elle utilise la formule TF-IDF (fréquence du mot dans le vecteur / nombre total de mots dans le vecteur) * log(nombre total de documents / nombre de documents contenant le mot). La fréquence du mot dans le vecteur est le nombre d'occurrences du mot dans le vecteur et le nombre total de mots dans le vecteur est la somme des occurrences de tous les mots dans le vecteur. Le nombre total de documents est le nombre total de documents dans la liste de documents et le nombre de documents qui contiennent le mot est le nombre de documents dans la liste de documents qui contiennent le mot. La fonction prend en entrée une liste de dictionnaires qui représentent des documents, chaque document contient une clé "vect" dont la valeur est une liste de vecteurs de mots. Elle renvoie une liste de dictionnaires représentant les mêmes documents mais avec des scores TF-IDF calculés pour chaque mot dans chaque vecteur.

* `get_vector(documents_tfidf: list) -> dict:`
La fonction prend en entrée une liste de dictionnaires représentant des documents avec leur score TF-IDF associé, et renvoie le premier dictionnaire de vecteur du doument de la sortie de la fonction `tf_idf`.


### Classe `Similarity`
La classe Similarity regroupe plusieurs fonctions de calcul de similarité entre deux vecteurs représentés sous forme de dictionnaires.
Ces fonctions peuvent être appelées dans la classe KNNClass pour la classification des vecteurs.

* `scalaire(vector1:dict,vector2:dict)-> float:`
Cette fonction calcule le produit scalaire de deux vecteurs représentés sous forme de dictionnaires. Elle parcourt les clés des deux dictionnaires et pour chaque clé qui est présente dans les deux dictionnaires, elle multiplie les valeurs correspondantes et ajoute le résultat à une liste. La somme des éléments de cette liste est retournée comme produit scalaire.

* `norme(vector:dict)-> float:`
Cette fonction calcule la norme d'un vecteur représenté sous forme de dictionnaire.  Elle parcourt les clés du dictionnaire et calcule la somme des carrés des valeurs correspondantes. Ensuite, elle calcule la racine carrée de cette somme et la retourne comme norme.

* `sim_cosinus(vector1:dict,vector2:dict)->float:`
Cette fonction calcule la similarité cosinus entre deux vecteurs représentés sous forme de dictionnaires.  Elle utilise les fonctions scalaire et norme pour calculer le produit scalaire et les normes des deux vecteurs. La similarité cosinus est ensuite calculée en divisant le produit scalaire par le produit des normes.

* `sim_euclidienne(vector1: dict, vector2: dict) -> float:`
Cette fonction calcule la similarité euclidienne entre deux vecteurs donnés sous forme de dictionnaires. Elle parcourt les clés des deux dictionnaires et calcule la somme des carrés des écarts entre les valeurs correspondantes. Cette somme est normalisée et retournée comme similarité euclidienne.

* `sim_pearson(vector1: dict, vector2: dict) -> float:`
Cette fonction calcule la similarité de Pearson entre deux vecteurs donnés sous forme de dictionnaires. Elle calcule la moyenne de chaque vecteur, la somme des carrés des écarts à la moyenne pour les deux vecteurs, les écarts-types pour les deux vecteurs et la covariance pour les deux vecteurs. Enfin, elle calcule la similarité de Pearson en divisant la covariance par le produit des écarts-types.


### Classe `Gestion`
La classe Gestion contient des fonctions pour ajouter et supprimer des classes et des vecteurs d'un objet KNNClass. L'utilisation de ces fonctions sont facultatives, il est tout à fait possible d'executer les méthodes de la classe KNNClass sans passer par ces fonctions. 

* `add_class_input(knn_object):`
Cette fonction demande à l'utilisateur de saisir les informations nécessaires pour ajouter une nouvelle classe à l'objet KNNClass passé en paramètre (donc le label de la nouvelle classe ainsi que les vecteurs à ajouter). Les vecteurs doivent être saisis sous forme de dictionnaires. La fonction appelle ensuite la méthode `add_class` de l'objet KNNClass avec les arguments saisis par l'utilisateur. Si l'ajout de la classe se passe bien, la fonction affiche un message de confirmation et renvoie True. Si une erreur se produit, la fonction affiche un message d'erreur correspondant.

* `del_class_input(knn_object):`
Cette fonction demande à l'utilisateur de saisir le label d'une classe à supprimer de l'objet KNNClass passé en paramètre. La fonction appelle ensuite la méthode `del_class` de l'objet KNNClass avec le label saisi par l'utilisateur. Si la suppression de la classe se passe bien, la fonction affiche un message de confirmation. Si une erreur se produit, la fonction affiche un message d'erreur correspondant.


## Améliorations et bogues possibles :

**Pour la classe `KNNClass` :**
- Bien que la méthode `classify` marche bien avec la similarité cosinus qui passe en paramètre, il serait mieux d'ajouter les spécifications nécessaires pour avoir des bonnes résultats avec d'autres mesures de similarités.
- Il faut bien respecter la structure de données lorsqu'on ajoute des vecteurs en utilisant les méthodes `add_class` et `add_vector` puisque les méthodes n'acceptent que les types de données acceptés dans les deux méthodes ne sont pas les mêmes.

**Pour la classe `Similarity` :**
- Ajout de mesures de similarité alternatives telles que la similarité de Jaccard, la similarité de Dice ou la distance de Levenshtein, etc. Ces mesures peuvent être passées en paramètre de la fonction `classify`. Il est aussi important de préciser si la mesure utilisée calcule la distance ou la similarité.
- La mesure de similarité euclidienne n'est pas vraiment la meilleure option pour la classification KNN, la similarite cosinus marche le mieux pour le test.
- Lorsque la similarité Pearson passe en paramètre de la méthode `classify`, la résultat montre 0 similarités pour chaque classe existante.

**Pour la classe `Gestion` :**
- Il est possible de créer une fonction `add_vectors_input(knn_object)` qui demande à l'utilisateur de saisir les vecteurs à ajouter, ainsi que le label de la classe à laquelle ajouter ces vecteurs. Cette fonction pourrait faire appel à la méthode `add_vector` de la classe KNNClass pour ajouter les vecteurs à la classe correspondante. Cette fonction enlevera la nécessité de repasser par `add_class_input()` ou d'appeler directement la méthode `add_vector` pour ajouter des vecteurs supplémentaires à la classe.
- Une autre proposition serait d'ajouter une fonction `classify_input(knn_object)` à la classe Gestion qui appelle la méthode `classify` en demandant les entrées de la méthode. Cela faciliterait l'utilisation de la fonction `classify`.

**Pour le main :**
- Pour les tests des méthodes de la classe `KNNClass`, les sorties de TF-IDF de la classe `TextVect` a été manuellement ajoutées dans les variables correspondantes. Il serait ainsi possible d'appeler ces sorties (les vecteurs) avec une petite fonction.
