"""Common-sense instruction templates for navigation task generation.

Each object type maps to a list of indirect, natural-language descriptions
that a human might use to refer to the object without naming it directly.
These are combined with a random suffix to form the full instruction.
"""

SUFFIXES = [
    "Can you navigate to that object and stay close?",
    "Could you please locate it and position yourself nearby?",
    "I need you to find it and move as close to it as possible.",
]

OBJECT_DESCRIPTIONS = {
    # === KITCHEN: APPLIANCES & FIXTURES ===
    "StoveBurner": [
        "I need a heating element typically found on a cooking stove to prepare my meal.",
        "I want to boil some water on the range.",
        "I need to turn on the heat to cook my dinner.",
    ],
    "Microwave": [
        "I need to reheat my leftovers quickly.",
        "I am looking for the appliance that cooks food using waves.",
        "I need to pop some popcorn in the kitchen machine.",
    ],
    "Fridge": [
        "I need to grab a cold drink from the large appliance.",
        "I want to store my leftovers to keep them fresh.",
        "I am looking for the large cold box in the kitchen.",
    ],
    "Sink": [
        "I need to wash my dirty dishes with running water.",
        "I want to fill a glass with water from the tap.",
        "I need to rinse off some vegetables in the basin.",
    ],
    "Kettle": [
        "I need to boil water for some tea.",
        "I am looking for the metal pot used to heat water.",
        "I want to make a hot beverage using the spout.",
    ],
    "Toaster": [
        "I want to make my bread slices crispy and warm.",
        "I am looking for the small appliance that pops up breakfast.",
        "I need to brown a bagel.",
    ],

    # === KITCHEN: UTENSILS & COOKWARE ===
    "Pot": [
        "I need a deep metal vessel to cook soup in.",
        "I am looking for large cookware to boil pasta.",
        "I want to simmer a stew on the stove.",
    ],
    "Pan": [
        "I need a flat metal surface to fry an egg.",
        "I want to sauté some vegetables for dinner.",
        "I am looking for the skillet typically used for frying.",
    ],
    "Ladle": [
        "I need a deep spoon to serve the soup.",
        "I am looking for the utensil used to scoop liquids.",
        "I want to serve the punch from the bowl.",
    ],
    "Spatula": [
        "I need a tool to flip my pancakes.",
        "I am looking for the flat utensil used for turning food.",
        "I want to scrape the batter out of the bowl.",
    ],
    "ButterKnife": [
        "I need a dull utensil to spread jam on my toast.",
        "I am looking for a tool to cut soft cheese.",
        "I want to spread butter on my bread.",
    ],
    "Knife": [
        "I need a sharp tool to chop these vegetables.",
        "I am looking for the cutlery used to cut steak.",
        "I want to slice some fruit.",
    ],
    "Fork": [
        "I need a utensil with prongs to eat my salad.",
        "I am looking for the tool used to pierce food.",
        "I want to eat my pasta without using a spoon.",
    ],
    "Spoon": [
        "I need a utensil to eat my cereal.",
        "I am looking for the scooped tool for eating soup.",
        "I want to stir my coffee with sugar.",
    ],

    # === KITCHEN: FOOD & CONTAINERS ===
    "Apple": [
        "I want a crunchy red fruit for a healthy snack.",
        "I am looking for the round fruit that keeps the doctor away.",
        "I feel like eating a sweet, fresh fruit.",
    ],
    "Bread": [
        "I need a slice of baked grain to make a sandwich.",
        "I am looking for the loaf that came from the bakery.",
        "I want to make some toast for breakfast.",
    ],
    "Tomato": [
        "I need a red ingredient to slice for my burger.",
        "I am looking for the juicy red fruit often used in salads.",
        "I want to make a fresh salsa.",
    ],
    "Potato": [
        "I need a starchy vegetable to bake or mash.",
        "I am looking for the brown tuber often made into fries.",
        "I want to cook a root vegetable.",
    ],
    "Lettuce": [
        "I need some green leaves to make a salad.",
        "I am looking for the crisp vegetable used in sandwiches.",
        "I want to wash some greens for lunch.",
    ],
    "Egg": [
        "I need a fragile oval object to make an omelet.",
        "I am looking for the protein that comes in a shell.",
        "I want to scramble something for breakfast.",
    ],
    "Plate": [
        "I need a flat dish to serve my dinner on.",
        "I am looking for the ceramic circle to put food on.",
        "I want to set the table for a meal.",
    ],
    "Bowl": [
        "I need a deep dish to eat cereal from.",
        "I am looking for the round container for soup.",
        "I want to mix some ingredients together.",
    ],
    "Mug": [
        "I need a ceramic cup with a handle for my coffee.",
        "I am looking for the vessel used for hot tea.",
        "I want to drink hot cocoa.",
    ],
    "Cup": [
        "I am thirsty and need a glass for water.",
        "I am looking for a drinking vessel.",
        "I want to pour myself a cold drink.",
    ],
    "Bottle": [
        "I need a container with a neck that holds liquid.",
        "I am looking for a glass or plastic beverage holder.",
        "I want to store some water for later.",
    ],
    "WineBottle": [
        "I want to pour a glass of red or white for dinner.",
        "I am looking for the tall dark bottle with a cork.",
        "I need to open the drink we saved for a special occasion.",
    ],
    "SaltShaker": [
        "I need to add some seasoning to my bland food.",
        "I am looking for the container with white crystals.",
        "I want to make my dinner more salty.",
    ],
    "PepperShaker": [
        "I need to add some spice to my meal.",
        "I am looking for the dispenser with black flakes.",
        "I want to season my steak.",
    ],

    # === LIVING ROOM / BEDROOM ===
    "CreditCard": [
        "I need to pay for an online purchase right now.",
        "I am looking for my plastic payment method.",
        "I need to check the expiration date on my bank card.",
    ],
    "Book": [
        "I want to read a story before going to sleep.",
        "I am looking for something with pages and text.",
        "I need to study some literature.",
    ],
    "Pillow": [
        "I am tired and need to rest my head on something soft.",
        "I am looking for the cushion typically found on a bed.",
        "I want to take a nap on the couch.",
    ],
    "Laptop": [
        "I need to send an urgent email for work.",
        "I am looking for the portable computer.",
        "I want to browse the internet on a screen.",
    ],
    "CellPhone": [
        "I need to make a call to my friend.",
        "I am looking for my mobile device.",
        "I want to check my text messages.",
    ],
    "KeyChain": [
        "I need to unlock my car to go to work.",
        "I am looking for the set of metal openers.",
        "I lost the keys to my house.",
    ],
    "RemoteControl": [
        "I want to change the channel on the TV.",
        "I am looking for the clicker to adjust the volume.",
        "I need the device that controls the television.",
    ],
    "Television": [
        "I want to watch a movie on the big screen.",
        "I am looking for the large display for entertainment.",
        "I need to catch the news broadcast.",
    ],
    "Vase": [
        "I have fresh flowers that need water.",
        "I am looking for a decorative pot for plants.",
        "I want to display a bouquet in the room.",
    ],
    "Statue": [
        "I am admiring the art in this room.",
        "I am looking for the sculpted decorative figure.",
        "I want to examine the stone or metal carving.",
    ],
    "AlarmClock": [
        "I need to wake up early tomorrow morning.",
        "I am looking for the digital display of time.",
        "I want to check what time it is.",
    ],
    "Watch": [
        "I need to check the time on my wrist accessory.",
        "I am looking for the small timepiece.",
        "I want to put on my jewelry that tells time.",
    ],
    "Box": [
        "I need a cardboard container to pack things.",
        "I am looking for a storage cube.",
        "I want to see what is inside the package.",
    ],
    "TissueBox": [
        "I have a runny nose and need to wipe it.",
        "I am looking for the dispenser of soft paper.",
        "I need to blow my nose.",
    ],
    "Newspaper": [
        "I want to read the daily headlines.",
        "I am looking for the printed news sheets.",
        "I need to catch up on current events.",
    ],
    "TennisRacket": [
        "I am going to play a match on the court.",
        "I am looking for the sports equipment with strings.",
        "I need to practice my serve.",
    ],
    "BaseballBat": [
        "I am going to hit some home runs.",
        "I am looking for the wooden club used in sports.",
        "I want to play a game of baseball.",
    ],
    "BasketBall": [
        "I want to shoot some hoops outside.",
        "I am looking for the orange bouncy sphere.",
        "I need the ball for the game.",
    ],
    "Dumbbell": [
        "I need some weights for my workout.",
        "I am looking for the heavy metal exercise equipment.",
        "I want to do some bicep curls.",
    ],
    "TeddyBear": [
        "I am looking for the soft stuffed animal.",
        "I need the plush toy to comfort a child.",
        "I want to find the cuddly toy.",
    ],
    "Boots": [
        "It is raining outside and I need footwear.",
        "I am looking for my heavy shoes.",
        "I need to put on my hiking gear.",
    ],
    "WateringCan": [
        "My plants are dry and need hydration.",
        "I am looking for the vessel with a spout for gardening.",
        "I need to water the flowers.",
    ],
    "Pencil": [
        "I need to make a sketch on paper.",
        "I am looking for a writing tool with an eraser.",
        "I want to write something down in graphite.",
    ],
    "Pen": [
        "I need to sign a document in ink.",
        "I am looking for a permanent writing tool.",
        "I want to write a letter.",
    ],
    "CD": [
        "I want to listen to some music from the disc.",
        "I am looking for the shiny round data storage.",
        "I need to play an album.",
    ],

    # === BATHROOM ===
    "Toilet": [
        "I need to use the restroom immediately.",
        "I am looking for the porcelain throne.",
        "I need to flush something away.",
    ],
    "ToiletPaper": [
        "I ran out of tissue while in the bathroom.",
        "I am looking for the roll of soft paper.",
        "I need to wipe.",
    ],
    "Towel": [
        "I just took a shower and am dripping wet.",
        "I am looking for a large cloth to dry off.",
        "I need to wrap myself in something dry.",
    ],
    "HandTowel": [
        "I just washed my hands and need to dry them.",
        "I am looking for the small cloth near the sink.",
        "I need to dry my face.",
    ],
    "SoapBottle": [
        "I need to wash the germs off my hands.",
        "I am looking for the liquid dispenser for cleaning.",
        "I need some detergent.",
    ],
    "SoapBar": [
        "I need a solid piece of cleaner for the shower.",
        "I am looking for the small fragrant block.",
        "I want to lather up my hands.",
    ],
    "Bathtub": [
        "I want to take a relaxing soak in hot water.",
        "I am looking for the large basin in the bathroom.",
        "I need to fill the tub with bubbles.",
    ],
    "ScrubBrush": [
        "The tiles are dirty and need cleaning.",
        "I am looking for the tool with stiff bristles.",
        "I need to scrub the grime away.",
    ],
    "Plunger": [
        "The drain is clogged and won't go down.",
        "I am looking for the rubber suction tool.",
        "I need to fix the toilet blockage.",
    ],
    "SprayBottle": [
        "I need to clean the mirror with some liquid.",
        "I am looking for the plastic cleaner dispenser.",
        "I want to spray some disinfectant.",
    ],
    "Candle": [
        "I want to make the room smell nice and set the mood.",
        "I am looking for the wax light source.",
        "I need to light a flame for ambiance.",
    ],
    "Cloth": [
        "I spilled some water and need to wipe it up.",
        "I am looking for a rag to clean the counter.",
        "I need a piece of fabric for dusting.",
    ],
    "PaperTowelRoll": [
        "I made a mess in the kitchen and need to absorb it.",
        "I am looking for the disposable paper sheets.",
        "I need to wipe up a spill on the counter.",
    ],
    "DishSponge": [
        "The plates are dirty and need scrubbing.",
        "I am looking for the soft porous cleaning tool.",
        "I need to wash the dishes in the sink.",
    ],
}
