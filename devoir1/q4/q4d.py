if __name__ == '__main__':
    # essayer python3 -m q4.q4d
    from q3.utils import Traceur
    from q4.q4c import CustomBayes

    clf = CustomBayes(cout_lambda=0.4)
    Traceur(avec_rejet=True).tracer_tous_les_duos(clf, show=True)
