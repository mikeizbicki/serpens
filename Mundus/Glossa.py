from lark import Lark

# colors: <https://www.reddit.com/r/latin/comments/11a5rk9/colors_in_latin_an_infographic/>

variables = {
    'case': ['nominative', 'accusative', 'dative', 'vocative'],
    'gender': ['masculine', 'feminine'],
    'number': ['singular', 'plural'],
    'time': ['present', 'past'],
    }

grammar_metalark = r'''
start: vp

vp: subj<number> verb<number>
    | verb<number> subj<number>

subj<number>: adj<case=nominative,gender,number>* noun<case=nominative,gender,number> adj<case=nominative,gender,number>*
subj<number=plural>: subj "et" subj

noun<declension=2,case=nominative,gender=masculine,number=singular>: /\w*us/
noun<declension=2,case=nominative,gender=masculine,number=plural>: /\w*i/

noun<declension=1,case=nominative,gender=masculine,number=singular>: /\w*a/
noun<declension=1,case=nominative,gender=masculine,number=plural>: /\w*ae/

adj<declension=12,case=nominative,gender=masculine,number=singular>: root ~ 'us'
adj<declension=12,case=nominative,gender=masculine,number=plural>: root ~ 'i'

verb<declension=are,time=present,number=singular>: /\w*at/
verb<declension=are,time=present,number=plural>: /\w*ant/

%import common.WS
%ignore WS
'''

'''

noun_phrase<Case a,Gender b,Number c>: noun<Case a,Gender b,Number c>* adj<Case a,Gender b,Number c>* noun_phrase<Case 'genativus',gender b2,number=c2>*

noun<declension '1',case 'nominative',gender 'masculine',number 'singular'>: /\w*a/
noun<declension '1',case 'nominative',gender 'masculine',number 'plural'>: /\w*ae/

verb<declension '1',time 'present',number 'singular'>: /\w*at/
verb<declension '1',time 'present',number 'plural'>: /\w*ant/
'''

def metaparse(text):
    '''
    >>> metaparse("verb[declension '1',time 'present',number 'plural']: /\w*at/")
    verb__declension_1_time_present_number_plural: /\w*at/
    '''
    lexpr, rexpr = text.split(':')

grammar = r'''
start: vp | vp vp

vp: subj_singular verb_singular
    | verb_singular subj_singular
    | verb_plural subj_plural
    | subj_plural verb_plural 

subj_singular: adj_nom_masculine_singular* noun_nom_masculine_singular adj_nom_masculine_singular*

subj_plural: subj_singular "et" subj_singular
    | adj_nom_masculine_plural* noun_nom_masculine_plural adj_nom_masculine_plural*

noun_nom_masculine_singular: noun_deci_nom_masculine_singular
    | noun_decii_nom_masculine_singular

noun_deci_nom_masculine_singular: /\w*a/
noun_deci_nom_masculine_plural: /\w*ae/
noun_decii_nom_masculine_singular: /\w*us/
noun_decii_nom_masculine_plural: /\w*i/
adj_deci_nom_masculine_singular: /\w*us/
adj_deci_nom_masculine_plural: /\w*i/

verb_singular: /\w*at/
verb_plural: /\w*ant/

%import common.WS
%ignore WS
'''



if __name__ == '__main__':
    parser = Lark(grammar, ambiguity='explicit')
    texts = [
        'inimicus albinus ambulat',
        'ambulat inimicus',
        'oppugnat inimicus',
        'oppugnant inimicus et amicus',
        #'accipe pecuniam'
        #'accipe rem'
        ]
    for text in texts:
        print(f"text={text}")
        p = parser.parse(text)
        print(f"p.pretty()={p.pretty()}")
