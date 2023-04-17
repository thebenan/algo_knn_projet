import re
import math
import copy
import json
import os
from typing import List, Tuple



class KNNClass:
  
    def __init__(self, description: str="", data=[]):
        """
        initialisation de la classe
        Args:
        - description (str): chaîne de caractères décrivant l'ensemble de classes
        - data (list): une liste contenant les vecteurs associés à chaque classe
        """
        self.description = description
        self.data =  data

    def add_class(self, label: str, vectors: list):
        """
        ajoute une nouvelle classe avec le label et les vecteurs fourni
        Args:
        - label (str): le nom de la classe à ajouter
        - vectors (list): une liste de vecteurs à associer à la classe
        Raises:
        - ValueError: si la classe avec le label fourni existe déjà dans le modèle
        """
        if any(cls['label'] == label for cls in self.data):
            raise ValueError(f"La classe {label} existe déjà dans le modèle")
        else:
            new_class = {'label': label, 'vect': vectors}
            self.data.append(new_class)
            print("Classe ",label," a été ajoutée en utilisant la fonction add_class")

        
    def get_classes(self):
        """
        retourne un dictionnaire contenant les classes et leurs vecteurs
        """
        return list(self.data.copy())
        

    def add_vector(self, label: str, vector):
        """
        ajoute un vecteur à une classe existante définie par le label fourni
        Args:
        - label (str): le nom de la classe où on ajoute le vecteur
        - vector : le vecteur à ajouter
        Raises:
        - ValueError: si la classe avec le label fourni n'existe pas dans le modèle
        """
        for i in range(len(self.data)):
            if self.data[i]["label"] == label:
                self.data[i]["vect"].append(vector)
                print("Vecteur ajouté à la classe", label," en utilisant la fonction add_vector")
                return
        raise ValueError(f"La classe {label} n'existe pas dans le modèle")


    def del_class(self, label:str):
        """
        supprime la classe correspondant au label fourni
        Args:
        - label (str): le nom de la classe à supprimer
        Raises:
        - ValueError: si la classe avec le label fourni n'existe pas dans le modèle
        """
        for cls in self.data:
            if cls['label'] == label:
                self.data.remove(cls)
                print(f"La classe {label} a été supprimée en utilisant la fonction del_class")
                return
        raise ValueError(f"La classe {label} n'existe pas dans le modèle")



    def save_as_json(self, filename:str):
        """
        enregistre les données de la classe actuelle au format JSON dans un fichier
        Args:
        - filename (str): le nom du fichier où on enregistre les données
        Output:
        - False en cas d'erreur d'entrée/sortie, sinon None
        """
        try:
            with open(filename, 'w') as outfile:
                json.dump({'description': self.description, 'data': self.data}, outfile, ensure_ascii=False)                
        # ce bloc s'execute en cas d'erreur
        except IOError as err:
            print("Impossible d'ouvrir",filename,f"Erreur={err}")
            return False
        print("Le fichier JSON", filename," a été enregistré en utilisant la fonction save_as_json")

    def load_as_json(self, filename:str):
        """
        charge les données depuis un fichier JSON et remplace la classe actuelle par les données chargées
        Args:
        - filename (str): le nom du fichier contenant les données à charger
        Output:
        - False : en cas d'erreur d'entrée/sortie, sinon None
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.description = data['description']
                self.data = data['data']
        except IOError as err:
            print("Impossible d'ouvrir", filename, f"Erreur: {err}")
            return False
        print("Le fichier JSON", filename," a été chargé en utilisant la fonction save_as_json")


    def classify(self, vector: dict, k: int, sim_func=None) -> List[Tuple[str, float]]:
            """
            cette fonction récupère un vecteur sous forme de hashage 
            le nombre de voisins les plus proches à considérer, et une fonction de similarité sim_func
            et renvoie une liste triée de paires [label:str,sim:float] pour les classes candidates
            la liste est triée par similarité décroissante, la similarité étant la moyenne des similarités
            obtenues sur les vecteurs retenus pour la classe correspondante dans les k plus proches voisins
            par défaut sim_func est le calcul de cosinus

            Input :
                arg1 : vector - hash
                arg2 : k - int
                arg3 : sim_func - function, par défaut : sim_cosinus
            Output :
                valeur de retour : une liste triée de paires [label:str,sim:float] - List[Tuple[str,float]]
            """
            # on utilise par défaut la fonction sim_cosinus si aucune fonction de similarité n'est pas fournie
            if sim_func is None:
                sim_func = Similarity.sim_cosinus
            try:
                # dictionnaire pour stocker les similarités moyennes pour chaque label
                sim_dict = {}
                # parcours de chaque dictionnaire dans la liste des dictionnaires des vecteurs
                for data_dict in self.data:
                    # récupère le label du dictionnaire courant
                    label = data_dict['label']
                    # liste pour stocker les similarités pour chaque vecteur du label
                    sim_list = []        
                    # parcours de chaque vecteur du label, si le label n'a pas de vecteurs, on saute cette étape
                    if len(data_dict['vect']) == 0:
                        continue
                    for vec in data_dict['vect']:
                        # calcul de la similarité entre le vecteur d'entrée et le vecteur actuel du label
                        sim_list.append(sim_func(vector, vec))
                    # tri décroissant des similarités
                    sim_list.sort(reverse=True)
                    # calcul de la similarité moyenne pour les k plus proches voisins
                    sim_dict[label] = sum(sim_list[:k])/k
                # tri décroissant des labels par similarité moyenne
                sorted_sim = sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)
                # retourne la liste triée de paires [label:str,sim:float]
                return sorted_sim
            except Exception as e:
                print(f"Une erreur s'est produite lors de l'exécution de la fonction classify: {e}")
                return False














class TextVect:


    # tokenize à partir du regex (dans read_texts)
    def tokenize(text:str,tok_grm)->list:
        """
        Input : 
            arg1 : un texte à tokeniser
            arg2 : une regex compilée pour la tokenisation
        Output : 
            valeur de retour : la liste des tokens
        """
        # normalisation des apostrophes pour une traitement correcte
        text = re.sub(r"’", "'", text)
        # normalisation pour oe en œ
        text = re.sub(r"œ", "oe", text)
        return tok_grm.findall(text)

    # vectorise
    def vectorise(tokens:list)->dict:
        """
        Input :
            arg1 : tokens - list(str)
        Output :
            valeur de retour : un dict contenant les vocables (clés) et les fréq associées (valeur)
        """
        token_freq={}  # initialisation du hachage
        for token in tokens: 
            if token not in token_freq.keys(): 
                token_freq[token]=0 
            token_freq[token]+=1 #on associe la fréquence à la clé (token)
        return token_freq
 
    
    def read_texts(folder_names: list) -> list:
        """
        lit tous les fichiers texte présents dans les dossiers de la liste folder_names
        tokenise les textes et crée des vecteurs pour chaque fichier
        retourne une liste de dictionnaires chacun contenant le label du dossier et une liste de vecteurs
        
        Input:
            folder_names (list): liste des noms de dossiers à parcourir pour lire les fichiers
        
        Output:
            list: liste de dictionnaires contenant le label et les vecteurs de chaque fichier
        """
        
        tok_grm = re.compile(r"""
            (?:etc.|p.ex.|cf.)| # quelques abréviations courantes
            \w+(?=(?:-(?:je|tu|ils?|elles?|nous|vous|leur|lui|les?|ce|t-|même|ci|là|y)))| # pour capturer les mots composés avec un tiret
            [\w\-]+'?'| # pour capturer les mots avec des apostrophes
            [^\W\d]+ # pour capturer seulement des lettres sans des chiffres 
                    # comme mes données sont des recettes, il est préferable d'ignorer les chiffres 
        """, re.X)

        vectors = []
        # parcours des dossiers dans "./KNN_corpus"
        for folder_name in folder_names:
            try:
                file_names = [f for f in os.listdir("./KNN_corpus/" + folder_name) if os.path.isfile("./KNN_corpus/" + folder_name + "/" + f)]
            except IOError as err:
                print("Impossible de lire les fichiers dans le dossier", folder_name, f"Erreur: {err}")
                continue # passe à la deuxième boucle 
            vector = []
            # parcours des fichiers dans le dossier
            for file_name in file_names:
                try:
                    # ouverture en lecture
                    my_dir = './KNN_corpus/' + folder_name + '/' + file_name
                    input_file = open(my_dir, mode="r", encoding="utf8")
                except IOError as err:
                    print("Impossible d'ouvrir", file_name, f"Erreur: {err}")
                    continue
                tokens = []
                for line in input_file:
                    line = line.strip()
                    toks = TextVect.tokenize(line, tok_grm)  # tokenisation des lignes
                    tokens.extend([tok.lower() for tok in toks]) # conversion en minuscule des tokens
                input_file.close()
                # ajout du vecteur correspondant au fichier dans la liste vector
                vector.append(TextVect.vectorise(tokens))
            # ajout du dictionnaire contenant le label et les vecteurs des fichiers dans la liste vectors
            vectors.append({'label': folder_name, 'vect': vector})
#        print(vectors)
        return vectors


    def read_txt(file_name:str) -> list:
        """
        lit tous les fichiers texte présents dans les dossiers de la liste folder_names
        tokenise les textes et crée des vecteurs pour chaque fichier
        retourne une liste de dictionnaires chacun contenant le label du dossier et une liste de vecteurs

        Input:
            folder_names (list): liste des noms de dossiers à parcourir pour lire les fichiers

        Output:
            list: liste de dictionnaires contenant le label et les vecteurs de chaque fichier
        """

        tok_grm = re.compile(r"""
            (?:etc.|p.ex.|cf.)| # quelques abréviations courantes
            \w+(?=(?:-(?:je|tu|ils?|elles?|nous|vous|leur|lui|les?|ce|t-|même|ci|là|y)))| # pour capturer les mots composés avec un tiret
            [\w\-]+'?'| # pour capturer les mots avec des apostrophes
            [^\W\d]+ # pour capturer seulement des lettres sans des chiffres 
                    # comme mes données sont des recettes, il est préferable d'ignorer les chiffres 
        """, re.X)

        vector = []
        vectors =[]
        # parcours du fichier
        try:
            # ouverture en lecture
            input_file = open(file_name, mode="r", encoding="utf8")
            tokens = []
            for line in input_file:
                line = line.strip()
                toks = TextVect.tokenize(line, tok_grm)  # tokenisation des lignes
                tokens.extend([tok.lower() for tok in toks]) # conversion en minuscule des tokens
            # ajout du vecteur correspondant au fichier dans la liste vector
            vector.append(TextVect.vectorise(tokens))
            # ajout du dictionnaire contenant le label et les vecteurs des fichiers dans la liste vectors
            vectors.append({'label': file_name, 'vect': vector})
        except IOError as err:
            print("Impossible d'ouvrir", file_name, f"Erreur: {err}")
        finally :
            input_file.close()

        return vectors


    # read_dict
    def read_dict(stoplist_filename):
        """
        Lecture d'une stoplist à partir d'un fichier
        Input : 
        arg : str - nom du fichier à lire. Un mot par ligne.
        Output :
        valeur de retour : set(str) - ensemble des stopwords
        """
        # on ouvre, lit et après ferme le fichier
        dict_file = open(stoplist_filename, "r", encoding="utf8")
        dict_content = dict_file.read()
        dict_file.close()
        # on sépare le dict_content(string) avec la saut de ligne et renvoie une liste
        stoplist = set(dict_content.split("\n"))
#        print(stoplist)
        return stoplist

    def filtrage(stoplist:set, documents, non_hapax:bool)->list:
        """
        A partir d'une liste de documents (objets avec deux propriétés 'label' et 'vect')
        on élimine tous les vocables appartenant à la stoplist.
        Input :
        arg1 : set - l'ensemble des stopwords
        arg2 : list(doc) - un doc est un dict contenant deux clés : 'label' et 'vect'
                doc : { 'label':str, 'vect':dict }
        arg3 : bool - indique si on veut éliminer les hapax (True) ou non (False)
        Output :
            list : les documents filtré
        """
        # on crée une liste vide comme une liste de documents avec des tokens filtrés
        documents_filtre = []
        for document in documents:
            # on crée un nouveau document filtré avec la même propriété 'label'
            document_filtre = {"label": document["label"]}
            #liste de dict de vecteurs
            document_filtre["vect"] = []
            for tokens in document["vect"]:
                # dict vect
                tokens_filtre = {}
                for token, freq in tokens.items():
                # on filtre les tokens en fonction de la stoplist et de la fréquence des mots
                    if token.lower() not in stoplist:
                        if non_hapax:
                            if freq > 1:
                                tokens_filtre[token] = freq
                        else:
                            tokens_filtre[token] = freq
                # on ajoute les tokens filtrés au document filtré
                document_filtre["vect"].append(tokens_filtre)
            # on ajoute le document filtré à la liste des documents filtrés
            documents_filtre.append(document_filtre)             
        # on retourne la liste des documents filtrés
        return documents_filtre


    def tf_idf (documents:list)->list:
        """
        calcul du score TF-IDF pour chaque mot dans chaque vecteur de chaque document d'une liste de documents
        TF-IDF = (fréquence du mot dans le vecteur / nombre total de mots dans le vecteur) * log(nombre total de documents / nombre de documents contenant le mot)
            - la fréquence du mot dans le vecteur est le nombre d'occurrences du mot dans le vecteur
            - le nombre total de mots dans le vecteur est la somme des occurrences de tous les mots dans le vecteur
            - le nombre total de documents est le nombre total de documents dans la liste de documents
            - le nombre de documents contenant le mot est le nombre de documents dans la liste de documents qui contiennent le mot
        Input : 
        arg1 : list(dict) : liste de dictionnaires représentant des documents 
                chaque document doit contenir une clé "vect" 
                dont la valeur est une liste de vecteurs de mots
        Output : 
            valeur de retour : list : liste de dictionnaires représentant les mêmes documents
                mais avec des scores TF-IDF calculés pour chaque mot dans chaque vecteur
        """
        # copie de la liste des documents pour éviter de modifier les données d'origine
        documents_new = copy.deepcopy(documents)
        # dictionnaire pour stocker le nombre de documents contenant chaque mot
        freq_doc = {}
        # itération sur chaque document dans la liste
        for document in documents_new:
            # liste de tous les mots présents dans les vecteurs du document
            all_words = set(  # crée un ensemble contenant tous les mots dans les vecteurs du document
                word  # pour chaque mot
                for vector in document["vect"]  # dans chaque vecteur du document
                for word in vector  # pour chaque mot dans le vecteur
            )
            # dict pour stocker le nombre d'occurrences de chaque mot dans le document
            word_count_dict = {}
            # itération sur chaque vecteur dans le document
            for vector in document["vect"]:
                # itération sur chaque mot dans le vecteur
                for word in vector:
                    # ajout du mot au dictionnaire et incrémentation de son nombre d'occurrences
                    if word not in word_count_dict:
                        word_count_dict[word] = 0
                    word_count_dict[word] += 1
            # calcul du score tf-idf pour chaque mot dans chaque vecteur dans le document
            for vector in document["vect"]:
                vector_word_count_dict = {}
                # calcul de la fréquence de chaque mot dans le vecteur
                for word in vector:
                    if word not in vector_word_count_dict:
                        vector_word_count_dict[word] = 0
                    vector_word_count_dict[word] += 1
                # normalisation des scores tf-idf pour chaque mot dans le vecteur
                for word in vector:
                    # calcul de la fréquence inverse du mot dans l'ensemble des documents
                    if word not in freq_doc:
                        freq_doc[word] = 0
                        for d in documents_new:
                            if word in set(w for v in d["vect"] for w in v):
                                freq_doc[word] += 1
                    idf = math.log(1 + len(documents_new) / freq_doc[word])
                    # calcul du score tf-idf pour le mot dans le vecteur
                    tf_idf_score = (vector_word_count_dict[word] / len(vector)) * idf
                    # stockage du score normalisé dans le vecteur
                    vector[word] = tf_idf_score
        return documents_new

  
    def get_vector(documents_tfidf: list) -> dict:
        """
        cette fonction renvoie le premier dictionnaire de vecteur de doc de la sortie de la fonction tf_idf

        Input:
        - documents_tfidf (list): une liste de dictionnaires représentant des documents 
                                  avec leur score TF-IDF associé

        Output:
        - dict: un dictionnaire de vecteur associé au premier doc de la liste
                si la liste et le premier doc existent et ont un vecteur associé
                sinon renvoie None
        """
        if not documents_tfidf:
            return None
        first_doc = documents_tfidf[0]
        if not first_doc['vect']:
            return None
        return first_doc['vect'][0]

   


    
    
    
    
class Similarity :
# classe des différentes similarités à appeler dans la fonction classify de la classe KNNClass


    def scalaire(vector1:dict,vector2:dict)-> float:
        """
        Cette fonction récupère deux vecteurs sous forme de hashage 
        et renvoie leur produit scalaire
        Input :
            arg1 : vector1 - hash
            arg2 : vector2 - hash
        Output :
            valeur de retour : un produit scalaire - float
        """
        liste_scalaire=[]
        for key in vector1:
            if key in vector2:
                liste_scalaire.append(vector1[key]*vector2[key])
        produit_scalaire=sum(liste_scalaire)
        return produit_scalaire

    def norme(vector:dict)-> float:
        """
        Cette fonction récupère un vecteur sous forme de hashage 
        et renvoie sa norme
        Input :
            arg1 : vector - hash
        Output :
            valeur de retour : une norme - float
        """
        norme_carre=0
        for key in vector:
            # print(key,"=>",vector[key])
            norme_carre+=vector[key]*vector[key]
        norme=math.sqrt(norme_carre)
        return norme

    def sim_cosinus(vector1:dict,vector2:dict)->float:
        """
        Cette fonction récupère deux vecteurs sous forme de hashage, 
        et renvoie leur cosinus
        en appelant les fonctions scalaire et norme
        Input :
            arg1 : vector1 - hash
            arg2 : vector2 - hash
        Output :
            valeur de retour : un cosinus - float
        """
        norme1=Similarity.norme(vector1)
        norme2=Similarity.norme(vector2)
        scal=Similarity.scalaire(vector1,vector2)
        cosinus=(scal/(norme1*norme2))
        return cosinus
    
    def sim_euclidienne(vector1: dict, vector2: dict) -> float:
        """
        calcul de la similarité euclidienne entre deux vecteurs
        la valeur de retour est un float compris entre 0 et 1
        Input :
            arg1 : vector1 - hash
            arg2 : vector2 - hash
        Output :
            valeur de retour : similarité euclidienne - float
        """
        # initialisation de la distance à 0
        distance = 0.0
        # parcours des clés des deux dictionnaires
        for k in vector1.keys():
            # calcul de la distance euclidienne pour chaque clé
            distance += (vector1[k] - vector2[k])**2
        # normalisation de la distance pour obtenir une valeur de similarité entre 0 et 1
        return 1/(1+math.sqrt(distance))
    

    def sim_pearson(vector1: dict, vector2: dict) -> float:
        """
        calcul de la similarité de Pearson entre deux vecteurs (mesure de la corrélation)
        varie entre -1 et 1 =>  1 indique une forte corrélation positive,
        -1 indique une forte corrélation négative et 0 indique l'absence de corrélation
        Input:
            arg1 vector1: premier vecteur - dict
            arg2 vector2: deuxieme vecteur - dict
        Output:
            valeur de retour : similarité de Pearson entre vector1 et vector2 - float
        """

        # calcul de la moyenne de chaque vecteur
        moy_vector1 = sum(vector1.values()) / len(vector1)
        moy_vector2 = sum(vector2.values()) / len(vector2)

        # calcul de la somme des carrés des écarts à la moyenne pour les deux vecteurs
        somme_carres_ecarts_vector1 = 0.0
        somme_carres_ecarts_vector2 = 0.0
        for k in vector1:
            if k in vector2:
                somme_carres_ecarts_vector1 += (vector1[k] - moy_vector1) ** 2
                somme_carres_ecarts_vector2 += (vector2[k] - moy_vector2) ** 2

        # calcul des écarts-types pour les deux vecteurs
        std_vector1 = math.sqrt(somme_carres_ecarts_vector1 / len(vector1))
        std_vector2 = math.sqrt(somme_carres_ecarts_vector2 / len(vector2))

        # calcul de la covariance pour les deux vecteurs
        cov = 0.0
        for k in vector1:
            if k in vector2:
                cov += (vector1[k] - moy_vector1) * (vector2[k] - moy_vector2)

        # calcul de la similarité de Pearson
        if std_vector1 == 0 or std_vector2 == 0:
            return 0
        else:
            return cov / (std_vector1 * std_vector2)




        
        
class Gestion :        
# fonctions de gestion des fonctions de la classe KNNClass



    def add_class_input(knn_object):
        """
        demande à l'utilisateur de saisir les données nécessaires pour exécuter la méthode add_class() de la classe KNNClass
        Input:
            - knn_object (KNNClass): l'objet KNNClass sur lequel exécuter la méthode add_class()
        """
        label = ""
        while not label:
            label = input("Entrez le label de la nouvelle classe : ")
            if not label:
                print("Erreur : le label ne peut pas être vide")

        vector_list = []
        vector_input = None
        while vector_input != '':
            vector_input = input("Entrez un vecteur (ou tapez entrée pour terminer) : ")
            print(f"vector_input = '{vector_input.strip()}'")
            if not vector_input:
                continue
            try:
                vector = eval(vector_input)  # convertit la chaîne de caractères en dictionnaire 
                vector_list.append(vector)
            except (NameError, SyntaxError):
                print("Erreur : vecteur invalide")

        print(f"Appel de la méthode add_class avec les arguments : label = {label}, vector_list = {vector_list}")
        try:
            knn_object.add_class(label, vector_list)
            print(f"Classe {label} ajoutée avec succès !")
        except ValueError as e:
            print(e)
        return True


        
    def add_vector_input(knn_object):
        """
        demande à l'utilisateur de saisir les données nécessaires pour exécuter la méthode add_vector() de la classe KNNClass
        Input:
            - knn_object (KNNClass): l'objet KNNClass sur lequel exécuter la méthode add_vector()
        """
        label = None
        while label is None:
            try:
                label = str(input("Entrez le label de la classe : "))
                if label not in knn_object.data:
                    print(f"La classe {label} n'existe pas encore dans le modèle")
                break
            except ValueError:
                print("Erreur : le label doit être un string")          
        vector = None
        vector_input = ''
        while vector_input.strip().lower() != 'q':
            vector_input = input("Entrez un vecteur (ou tapez 'q' pour terminer) : ")
            if vector_input.strip().lower() == 'q':
                break
            try:
                vector = eval(vector_input)  # convertit la chaîne de caractères en dictionnaire 
            except (NameError, SyntaxError):
                print("Erreur : vecteur invalide")            
            knn_object.add_vector(label, vector)
        print(f"Vecteur ajouté à la classe {label} avec succès !")
    
    
    def del_class_input(knn_object):
        """
        demande à l'utilisateur de saisir les données nécessaires pour exécuter la méthode del_class() de la classe KNNClass
        Input:
            - knn_object (KNNClass): l'objet KNNClass sur lequel exécuter la méthode del_class()
        """
        label = None
        while label is None:
            try:
                label = str(input("Entrez le label de la classe à supprimer : "))
                knn_object.del_class(label)
                print(f"La classe {label} a été supprimée avec succès !")
            except ValueError as e:
                print(e)


    # ajout d'une nouvelle classe à l'instance test_knn
#    add_class_input(test_knn)
#    add_vector_input(test_knn)

        
        
        
        
        
# main
if __name__ == "__main__":
    stoplist = TextVect.read_dict("stopwords_french.txt")
    folder_names=[f.name for f in os.scandir("./KNN_corpus") if f.is_dir()]
    textes = TextVect.read_texts(folder_names)
#    print(stoplist)
#    print(textes)
    filtered=TextVect.filtrage(stoplist, textes, False)
#    print(filtered)
    filtered_tfidf=TextVect.tf_idf(filtered)
#    print(filtered_tfidf)



    # instanciation de la classe KNNClass
    test_knn = KNNClass(description="données de test")
    
    # les données obtenues des fonctions de la classe TextVect :
    desserts_vectors1 = [{'pâte': 0.06301338005090412, 'farine': 0.06301338005090412, 'beurre': 0.041649578721552505, 'oeufs': 0.06301338005090412, 'pincée': 0.041649578721552505, 'sel': 0.03150669002545206, 'verres': 0.06301338005090412, 'eau': 0.041649578721552505, 'crème': 0.041649578721552505, 'pâtissière': 0.06301338005090412, 'lait': 0.06301338005090412, 'maïzena': 0.041649578721552505, 'cuillère': 0.041649578721552505, 'soupe': 0.03150669002545206, 'essence': 0.06301338005090412, 'café': 0.041649578721552505, 'sucre': 0.06301338005090412, 'jaunes': 0.06301338005090412, 'oeuf': 0.06301338005090412, 'glaçage': 0.06301338005090412, 'fondant': 0.06301338005090412, 'boîte': 0.041649578721552505}]    
    desserts_vectors2 = {'biscuit': 0.055451774444795626, 'fraisier': 0.055451774444795626, 'oeuf': 0.055451774444795626, 'entiers': 0.055451774444795626, 'farine': 0.055451774444795626, 'sucre': 0.055451774444795626, 'vanille': 0.055451774444795626, 'poudre': 0.055451774444795626, 'crème': 0.03665162927496621, 'mousseline': 0.055451774444795626, 'jaune': 0.055451774444795626, 'lait': 0.055451774444795626, 'litre': 0.055451774444795626, 'maïzena': 0.03665162927496621, 'beurre': 0.03665162927496621, 'doux': 0.055451774444795626, 'sirop': 0.055451774444795626, 'eau': 0.03665162927496621, 'décoration': 0.055451774444795626, 'fraise': 0.055451774444795626, 'lavée': 0.055451774444795626, 'équeutée': 0.055451774444795626, 'glace': 0.055451774444795626, 'pâte': 0.055451774444795626, 'amande': 0.055451774444795626}
    desserts_vectors3 = {'noisette': 0.055451774444795626, 'beurre': 0.03665162927496621, 'farine': 0.055451774444795626, 'maïzena': 0.03665162927496621, 'cuillère': 0.03665162927496621, 'soupe': 0.027725887222397813, 'cacao': 0.055451774444795626, 'oeufs': 0.055451774444795626, 'sucre': 0.055451774444795626, 'sirop': 0.055451774444795626, 'eau': 0.03665162927496621, 'ousse': 0.055451774444795626, 'vanille': 0.055451774444795626, 'ganache': 0.055451774444795626, 'pralinée': 0.055451774444795626, 'chocolat': 0.055451774444795626, 'praliné': 0.055451774444795626, 'type': 0.055451774444795626, 'pralinoise': 0.055451774444795626, 'noir': 0.055451774444795626, 'crème': 0.03665162927496621, 'liquide': 0.055451774444795626, 'cuilère': 0.055451774444795626, 'vermicelles': 0.055451774444795626, 'pralin': 0.055451774444795626}
    entrees_vectors1 = [{'mayonnaise': 0.07296286111157319, 'citron': 0.07296286111157319, 'pressé': 0.07296286111157319, 'thé': 0.07296286111157319, 'sauce': 0.07296286111157319, 'worcestershire': 0.07296286111157319, 'tabasco': 0.07296286111157319, 'goût': 0.07296286111157319, 'boîte': 0.04822582799337658, 'chair': 0.07296286111157319, 'crabe': 0.07296286111157319, 'branche': 0.07296286111157319, 'céleri': 0.07296286111157319, 'haché': 0.07296286111157319, 'échalote': 0.07296286111157319, 'hachée': 0.07296286111157319, 'avocats': 0.07296286111157319, 'piment': 0.04822582799337658, 'espelette': 0.04822582799337658}]
    entrees_vectors2 = {'betterave': 0.12602676010180824, 'feuilles': 0.12602676010180824, 'persil': 0.12602676010180824, 'soupe': 0.06301338005090412, 'moutarde': 0.12602676010180824, 'vinaigre': 0.12602676010180824, 'balsamique': 0.12602676010180824, 'huile': 0.08329915744310501, 'olive': 0.08329915744310501, 'sel': 0.06301338005090412, 'poivre': 0.08329915744310501}
    entrees_vectors3 = {'potimarron': 0.17328679513998632, 'tranches': 0.17328679513998632, 'jambon': 0.17328679513998632, 'parme': 0.17328679513998632, 'basilic': 0.17328679513998632, 'feuilles': 0.17328679513998632, 'sel': 0.08664339756999316, 'poivre': 0.11453634148426939}
    plats_vectors1 = [{'eau': 0.05090504065967528, 'riz': 0.07701635339554948, 'risotto': 0.07701635339554948, 'crevettes': 0.07701635339554948, 'roses': 0.07701635339554948, 'crème': 0.05090504065967528, 'fraîche': 0.07701635339554948, 'vin': 0.07701635339554948, 'blanc': 0.07701635339554948, 'oignons': 0.07701635339554948, 'parmesan': 0.07701635339554948, 'cube': 0.07701635339554948, 'bouillon': 0.07701635339554948, 'volaille': 0.07701635339554948, 'huile': 0.05090504065967528, 'olive': 0.05090504065967528, 'cuillères': 0.07701635339554948, 'soupe': 0.03850817669777474}]
    plats_vectors2 = {'filets': 0.11552453009332421, 'poulet': 0.11552453009332421, 'tomates': 0.11552453009332421, 'pommes': 0.11552453009332421, 'terre': 0.11552453009332421, 'concentré': 0.11552453009332421, 'oignon': 0.11552453009332421, 'piment': 0.07635756098951292, 'espelette': 0.07635756098951292, 'pincée': 0.07635756098951292, 'sel': 0.057762265046662105, 'poivre': 0.07635756098951292}
    plats_vectors3 = {'veau': 0.06301338005090412, 'sauté': 0.06301338005090412, 'carottes': 0.06301338005090412, 'oignon': 0.06301338005090412, 'vin': 0.06301338005090412, 'muscat': 0.06301338005090412, 'beurre': 0.041649578721552505, 'ail': 0.06301338005090412, 'gousse': 0.06301338005090412, 'cube': 0.06301338005090412, 'bouillon': 0.06301338005090412, 'volaille': 0.06301338005090412, 'huile': 0.041649578721552505, 'olive': 0.041649578721552505, 'cuillère': 0.041649578721552505, 'soupe': 0.03150669002545206, 'maïzena': 0.041649578721552505, 'café': 0.041649578721552505, 'bouquet': 0.06301338005090412, 'garni': 0.06301338005090412, 'sel': 0.03150669002545206, 'poivre': 0.041649578721552505}
    
    # execution des fonctions
    test_knn.add_class('desserts',desserts_vectors1)
    print(KNNClass.get_classes(test_knn))
    test_knn.del_class('desserts')
    print(KNNClass.get_classes(test_knn))
    test_knn.add_class('desserts',desserts_vectors1)
    test_knn.add_vector('desserts',desserts_vectors2)
    test_knn.add_vector('desserts',desserts_vectors3)
    test_knn.add_class('entrees',entrees_vectors1)
    test_knn.add_vector('entrees',entrees_vectors2)
    test_knn.add_vector('entrees',entrees_vectors3)
    test_knn.add_class('plats',plats_vectors1)
    test_knn.add_vector('plats',plats_vectors2)
    test_knn.add_vector('plats',plats_vectors3)
#    print(KNNClass.get_classes(test_knn))
    
    # enregistrement des données du modèle dans un fichier JSON
    test_knn.save_as_json("mydata_knn.json")
    
    test2_knn = KNNClass(description="test 2")
    # chargement des données du fichier "mydata_knn.json" dans l'instance test2_knn
    test2_knn.load_as_json("mydata_knn.json")
#    print(KNNClass.get_classes(test2_knn))


    knntext = TextVect.read_txt("riz_cantonais.txt")
    print(knntext)

    filteredknn=TextVect.filtrage(stoplist, knntext, False)
    print(filteredknn)
    filtered_tfidfknn=TextVect.tf_idf(filteredknn)
    print(filtered_tfidfknn)
    
    print(TextVect.get_vector(filtered_tfidfknn))


    # TEST classify
    datatestknn = {'dés': 0.046209812037329684, 'épaule': 0.046209812037329684, 'petits': 0.046209812037329684, 'pois': 0.046209812037329684, 'crevettes': 0.046209812037329684, 'cuites': 0.046209812037329684, 'décortiquées': 0.046209812037329684, 'oignon': 0.046209812037329684, 'blanc': 0.046209812037329684, 'riz': 0.046209812037329684, 'champignons': 0.046209812037329684, 'blancs': 0.046209812037329684, 'boîte': 0.046209812037329684, 'ciboulette': 0.046209812037329684, 'soupe': 0.046209812037329684}
    res = test_knn.classify(datatestknn,3, None)
    print(res)

