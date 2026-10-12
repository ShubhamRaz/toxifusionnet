"""Small built-in lexicons (seed lists). They are deliberately compact and meant to be EXTENDED for real experiments.

* ROMAN_HINDI / ENGLISH_COMMON drive the heuristic token-level language identifier.
* POSITIVE / NEGATIVE feed the sarcasm cue features (sentiment-contrast signal).
* ABUSE / THREAT are seed lexicons for the keyword baseline and weak severity labels.
"""

ROMAN_HINDI = set("""
hai hain ho hu hoon tha thi the thin hota hoti hote hoga hogi honge kya kyu kyun kyon kaise kaisa kaisi kab kaha kahan
kaun kisko kisne kitna kitni kitne kuch koi sab sabhi sabko woh wo vo yeh ye yah isse usse inse unse isko usko inko unko
main mai mein mujhe mujhko mera meri mere hum humko hamara hamari hamare tum tumhe tumko tumhara tumhari tumhare tera
teri tere tu tujhe tujhko aap aapko aapka aapki aapke apna apni apne ka ki ke ko se pe tak ne bhi hi toh nahi nahin na
mat aur lekin magar kyunki isliye jab tab agar warna phir fir abhi ab yahan wahan yaha waha idhar udhar aise waise jaise
jaisa vaise accha acha achha bura bahut bohot bhot zyada jyada kam thoda thodi bilkul sirf sach jhooth galat sahi theek
thik log logon logo aadmi aurat ladka ladki ladkiyan ladke bhai behen yaar dost insaan desh gaon shehar ghar kaam paisa
paise din raat subah shaam kal aaj saal baar baat baatein dekho dekh dekha dekhna bol bolo bola bolna bolte kar karo kiya
karna karte karta karti karenge karega kare jao jaa jana jaata jaate gaya gayi gaye aao aaya aayi aaye lo liya lena lete
diya dena dete rakh rakho raho rehna rehte reh chal chalo chalta chalte mil mila milega mile samajh samjha samajhna samjho
sun suno suna sunna soch socho sochna pata maloom chahiye chahte chahta chahti sakta sakte sakti paaye paya paana wala
wali wale waale liye lie bina saath sath baad pehle pehla andar bahar upar neeche kabhi hamesha hona hone hua hui hue wah
waah arre arey oye bas bass haan han ji zaroor shukriya dhanyavaad mazaa maza mast badhiya zabardast bekaar bekar ghatiya
bewakoof pagal nikamma nikamme gadha gande ganda gandi jaati jati dharm mazhab musalman hindu log unki unka uske uska
unke inke inka inki iske iska tumhari kisi kisiko sabka sabki sabke wahi yahi kyaa hamesha shayad lagta lagti lagte
bhaiya didi beta beti papa mama sasta mehnga naya purana bada badi bade chhota chhoti chhote pura puri poora
""".split())

# tokens that exist in both vocabularies; resolved with neighbour context
AMBIGUOUS = {"is", "in", "me", "to", "do", "par", "us", "the", "are", "na", "hi", "ho", "ha", "be", "so", "or"}

ENGLISH_COMMON = set("""
the a an and or but if so because for of in on at to from with without by about as is are was were be been being am have
has had do does did will would can could should may might must not no yes i you he she it we they me him her us them my
your his its our their this that these those what which who whom when where why how all any some more most other such only
own same than too very just also now then here there up down out over under again once great good bad nice best worst
love hate people country life time day work thanks thank please really truly obviously sure right wrong think know see
like want need make get go come say look wow lol lmao rofl omg genius brilliant amazing wonderful perfect idiot stupid
dumb fool trash garbage useless pathetic loser awesome fantastic congrats congratulations future bro dude guys man woman
women men girls boys government media news fake true lies truth real hope really always never every everyone nobody
""".split())

ENGLISH_SUFFIXES = ("ing", "tion", "sion", "ment", "ness", "ly", "ed", "ity", "ous", "able", "ful", "less")
HINDI_SUFFIXES = ("enge", "ega", "egi", "ayenge", "wala", "wali", "wale", "waala", "iyan", "iyon", "iye")

POSITIVE = set("""
great nice wow wonderful amazing brilliant genius perfect love best awesome fantastic thanks thank congrats
congratulations good lovely excellent superb wah waah wahh shabash badhiya zabardast mast mahaan mahan kamaal shandaar
lajawab dhanyavaad shukriya accha acha sahi sundar bahut_badhiya
""".split())

NEGATIVE = set("""
bad worst hate stupid idiot dumb fool trash garbage useless pathetic loser bekaar bekar ghatiya bewakoof pagal nikamma
nikamme gadha ganda gande gandi kamina kameena harami bakwas faltu jahil nalayak shame disgusting terrible awful
""".split())

SARCASM_MARKERS = [
    "kya baat", "wah ji wah", "waah waah", "oh really", "of course", "as if", "yeah right", "sure thing",
    "great job", "well done", "bahut khoob", "kitna accha", "kya kehne", "bilkul sahi", "ji haan", "shukriya",
    "thanks for", "genius", "brilliant", "mahaan", "kamaal", "how nice", "so smart", "kitne samajhdar",
]
LAUGH_RE = r"\b(?:l+o+l+|lmao+|rofl|ha(?:ha)+h*|he(?:he)+|hehe+|jk)\b"

POS_EMOJI = set("😀😃😄😁😆😊🙂😍👍👏🎉💯🙏")
NEG_EMOJI = set("😒🙄😏😡🤬👎🤡🤮😤😠💩")
LAUGH_EMOJI = set("😂🤣😹")
SARC_EMOJI = set("🙄😏🤡😒")

# Seed abuse lexicon for keyword baseline / weak severity labelling (extend for real use).
ABUSE = set("""
chutiya chutiye chutiyo chutiyapa madarchod behenchod bc mc bsdk gandu harami haramkhor kamina kameena kutta kutte kutiya
randi saala saale bhosdike lodu gaand chakka bhadwa bhadwe jahil nalayak suar idiot stupid moron trash scum dumb loser
bastard bitch asshole
""".split())

THREAT_PHRASES = [
    "maar dunga", "maar denge", "maar do", "jaan se", "kaat dunga", "kaat denge", "khatam kar", "jala do", "jala denge",
    "udaa denge", "goli maar", "mita do", "bhaga do", "nikal do", "kill", "murder", "die",
]
