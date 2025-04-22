from pysentimiento import create_analyzer

analyzer = create_analyzer(task="sentiment", lang="es")

def analiza( txt: str ):
    ret = analyzer.predict ( txt )

    return vars ( ret )

#print (  analiza("Me encanta la idea de hacer un remake de Karate kid") )
#print (  analiza("A ver cuando por fin termina la serie de Cobra Kai de una vez") )
#print ( analiza("Estos tios de Netflix van a quemar una idea brutal") ) 
