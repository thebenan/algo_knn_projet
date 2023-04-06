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
        if label in self.data:
            raise ValueError(f"La classe {label} existe déjà dans le modèle")
        else:
            self.data[label] = vectors
            print("Classe ajoutée")
        

    def add_vector(self, label: str, vector):
        """
        ajoute un vecteur à une classe existante définie par le label fourni
        Args:
        - label (str): le nom de la classe où on ajoute le vecteur
        - vector : le vecteur à ajouter
        """
        if label not in self.data:
            self.data[label] = [vector]
        else:
            self.data[label].append(vector)
            print("Ajout effectué")

    def del_class(self, label:str):
        """
        supprime la classe correspondant au label fourni
        Args:
        - label (str): le nom de la classe à supprimer
        Raises:
        - ValueError: si la classe avec le label fourni n'existe pas dans le modèle
        """
        if label not in self.data:
            raise ValueError(f"La classe {label} n'existe pas")            
        else:
            del self.data[label]
            print("Suppression effectuée")

    def save_as_json(self, filename:str):
        """
        enregistre les données de la classe actuelle au format JSON dans le fichier spécifié
        Args:
        - filename (str): le nom du fichier où on enregistre les données
        Output:
        - False en cas d'erreur d'entrée/sortie, sinon None
        """
        try:
            with open(filename, 'w') as outfile:
                json.dump({'description': self.description, 'data': self.data}, outfile)                
        # ce bloc s'execute en cas d'erreur
        except IOError as err:
            print("Impossible d'ouvrir",filename,f"Erreur={err}")
            return False
        return True

    def load_as_json(self, filename:str):
        """
        charge les données depuis un fichier JSON et remplace la classe actuelle par les données chargées
        Args:
        - filename (str): le nom du fichier contenant les données à charger
        Output:
        - False : en cas d'erreur d'entrée/sortie, sinon None
        """
        try:
            with open(filename, 'r') as infile:
                data = json.load(infile)
                self.description = data['description']
                self.data = data['data']
        except IOError as err:
            print("Impossible d'ouvrir", filename, f"Erreur: {err}")
            return False
        return True


    def classify(vector: dict, k: int, sim_func=None) -> List[Tuple[str, float]]:
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
            sim_func = TextVect.sim_cosinus
        try:
            # dictionnaire pour stocker les similarités moyennes pour chaque label
            sim_dict = {}
            # parcours de chaque label dans le dictionnaire des vecteurs
            for label, vec_list in vector_dict.items():
                # liste pour stocker les similarités pour chaque vecteur du label
                sim_list = []        
                # parcours de chaque vecteur du label
                for vec in vec_list:
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
            folder_names (list): Liste des noms de dossiers à parcourir pour lire les fichiers
        
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
            vector =[]
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
        # on parcourt chaque dictionnaire dans les dictionnaires de liste et crée un dictionnaire vide en tant que dictionnaire nouveau filtré
        for document in documents:
            # on crée un nouveau document filtré avec la même propriété 'label'
            document_filtre = {"label": document["label"]}
            document_filtre["vect"] = []
            for tokens in document["vect"]:
                # on crée un dictionnaire nouveau token_filtre
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
        Calcul du TF.IDF pour une liste de documents
        Input : 
        arg1 : list(dict) : une liste de documents ...
        Output : 
        valeur de retour : une liste de documents avec une modification des fréq
        associées à chaque mot (on divise par le log de la fréq de documents)
        """
        documents_new=copy.deepcopy(documents)
        #création d'un dict contenant tous les mots de tous les docs
        mots=set()
        # 1. on crée l'ensemble de tous les mots
        # on parcours les documents
        for doc in documents:
            #pour chaque mot du doc étant dans notre vecteur doc
            #word = notre variable qui récupère chaque mot
            for word in doc["vect"]:
                mots.add(word)
        # 2. on parcourt tous les mots pour calculer la fréquence de doc de chacun
        freq_doc={}
        for word in mots:
            # on parcourt les documents
            for doc in documents:
                if word in doc["vect"]:
                    if word not in freq_doc:
                        freq_doc[word]=1
                    else :
                        freq_doc[word]+=1        
        # 3. on parcourt les docs mot par mot pour mettre à jour la fréquence
        for doc in documents_new:
            for word in doc["vect"]:
                doc["vect"][word]=doc["vect"][word] / math.log(1+freq_doc[word])
        return documents_new

        

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
        norme1=TextVect.norme(vector1)
        norme2=TextVect.norme(vector2)
        scal=TextVect.scalaire(vector1,vector2)
        cosinus=(scal/(norme1*norme2))
        return cosinus
    
    def sim_euclidienne(vector1: dict, vector2: dict) -> float:
        """
        calcule la similarité euclidienne entre deux vecteurs
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
        calcule la similarité de Pearson entre deux vecteurs (mesure de la corrélation)
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




if __name__ == "__main__":
    stoplist = TextVect.read_dict("stopwords_french.txt")
    folder_names=[f.name for f in os.scandir("./KNN_corpus") if f.is_dir()]
    textes = TextVect.read_texts(folder_names)
#    print(stoplist)
#    print(textes)
    filtered=TextVect.filtrage(stoplist, textes, False)
#    print(filtered)
    filtered_tfidf=TextVect.tf_idf(filtered)
    print(filtered_tfidf)
    
    test_knn = KNNClass(description="données de test")


