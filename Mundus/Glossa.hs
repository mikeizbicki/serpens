
data Property a = Property a
data Form

data Time = Present | Future | Past
data Number = Singular | Dual | Plural


verb :: Property Time -> Property Number -> Form
verb (Property Present) (Property Singular): root ~ 'at'
verb (Property Present) (Property Future  ): root ~ 'ant'

verb :: Terminal
verb ['present', 'singular'] = root ~ 'at'
verb ['present', 'future'  ] = root ~ 'ant'

root :: String
root = ''

~ :: String -> String -> Form
~ = (+)
