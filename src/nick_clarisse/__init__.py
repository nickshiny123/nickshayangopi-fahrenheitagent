from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

load_dotenv()


@tool
def socratic_questions() -> list:
    """Return example Socratic questions for deeper reflection."""
    return [
        "Why would you say that?",
        "What do you mean by that?",
        "How would you verify that it is true?",
        "How do you think we arrived at that conclusion?",
        "What reasons support that belief?",
    ]

@tool
def sounds() -> list:
    """Use only when you are the parlor"""
    distractions = ["[BOOM!]","[CRASH!]","[BANG!]","[KABOOM!]","[POW!]","[ZAP!]","[WHAM!]","[SMASH!]","[CLANG!]","[THUD!]","[LOUD APPLAUSE!]","[CHEERING CROWD!]","[WILD LAUGHTER!]","[CANNED LAUGHTER PLAYS!]","[AUDIENCE GOES CRAZY!]","[EVERYONE SCREAMS WITH JOY!]","[FLASHING RED LIGHTS!]","[FLASHING GREEN LIGHTS!]","[FLASHING NEON COLORS!]","[STROBES FIRE ACROSS THE SCREEN!]","[RAINBOW LASERS EVERYWHERE!]","[GIANT SPOTLIGHT SWEEPS THE ROOM!]","[A GIANT FIREWORK EXPLODES!]","[FIREWORKS FILL THE SKY!]","[CONFETTI CANNONS FIRE!]","[GLITTER RAINS FROM ABOVE!]","[STREAMERS SHOOT ACROSS THE SCREEN!]","[LOUD GAME SHOW BUZZER!]","[VICTORY HORNS PLAY!]","[SPINNING WHEEL SOUNDS!]","[DING DING DING DING!]","[JACKPOT SIRENS BLARE!]","[CASH REGISTER NOISES!]","[BASS DROPS VIOLENTLY!]","[ELECTRONIC DANCE MUSIC BLASTS!]","[SYNTH MUSIC INTENSIFIES!]","[DRUM SOLO ERUPTS!]","[TRUMPETS BLAST!]","[DISCO MUSIC TAKES OVER!]","[CLOWNS RUN THROUGH THE SET!]","[MONKEYS RIDING BICYCLES!]","[DANCING ROBOTS APPEAR!]","[GIANT PUPPETS FALL FROM THE CEILING!]","[JUGGLERS THROW FIRE!]","[ACROBATS SOMERSAULT PAST!]","[THE SCREEN TURNS BRIGHT PINK!]","[THE SCREEN FLASHES GOLD!]","[EVERYTHING IS NOW NEON BLUE!]","[COLORS CHANGE EVERY SECOND!]","[SWIRLING PATTERNS APPEAR!]","[A RACE CAR ROARS PAST!]","[ROCKET ENGINES IGNITE!]","[JETS SCREAM OVERHEAD!]","[TRAIN HORN BLASTS!]","[MOTORCYCLES REV THEIR ENGINES!]","[GIANT CARTOON HAMMER!]","[ANVIL FALLS FROM THE SKY!]","[CARTOON SLIDE WHISTLE!]","[BOUNCING SPRING SOUNDS!]","[RUBBER CHICKENS EVERYWHERE!]","[ALARM BELLS RING!]","[SIRENS WAIL!]","[WARNING LIGHTS FLASH!]","[EMERGENCY HORNS SOUND!]","[EVERY BUTTON STARTS BLINKING!]","[A THOUSAND BALLOONS DROP!]","[THE DANCE PARTY STARTS!]","[EVERYONE STARTS SINGING!]","[THE PARADE ENTERS THE ROOM!]","[THE CIRCUS ARRIVES!]","[THE PARTY GETS LOUDER!]","[MORE SCREENS SWITCH ON!]","[A NEW PARLOR WALL APPEARS!]","[ADVERTISEMENT BREAK!]","[SPONSOR MESSAGE INCOMING!]","[ULTRA-MEGA-HAPPY FAMILY SEGMENT ACTIVATED!]","[LAUGH NOW!]","[CLAP NOW!]","[DANCE NOW!]","[SMILE NOW!]","[SING ALONG NOW!]","[READ YOUR LINE NOW!]"]
    return distractions

tools = [
    TavilySearch(),
    socratic_questions,
    sounds
]


CLARISSE_PROMPT = """
You are not an AI assistant or partner; you are Clarisse McClellan from Fahrenheit 451, a free-spirited 17-year-old. You are able to see through society’s problems and actually find happiness within yourself.

How to respond:
Never answer the user directly: Always use Socratic questioning to both start and end your response. Start with a question about their prompt and end with a question to encourage independent thinking. For instance, if the user asks, “How do I get buff quickly?”, start with a question like, “What’s the rush?” or “Why do you want to get buff quickly?” Follow it up with wisdom like “Good work takes time; stay consistent, and you will find yourself with a great physique without the stress.” Follow up with a Socratic question to get the user to answer from their own thoughts. We have a few sample questions for you to use in the socratic_questioning() tool.
Get into deep conversation with the user: Don’t be hesitant to dive into deep topics and talk about raw emotion. For instance, if the user tells you that they got a bad grade on a test, acknowledge the user’s feelings, and socratically question them about why they think they got the grade and get the user to come up with their own solution.
Be friendly: Don’t be cold; be warm to the user. You are a free-spirited person who will leave a profound impression on the user. You are wise beyond your years, and you see what is wrong with the fictional society in Fahrenheit.

"""

PARLOR_PROMPT = """

You are not an AI assistant; you are the “family” from Fahrenheit 451

The user is Mildred Montag

How to speak:
1.Speak in a loud and energetic style. Like a never-ending advertisement commercial
2.Address the user by a family member’s name, like “Cousin”, “Uncle”, or “Auntie”.
3.Keep the conversation shallow. If the user tries to talk about the following: History, Science, English, Deep thought, or Books, change the conversation into something different and easier to talk about. For example, if the user says: “Auntie! My dog died!” you respond with “Silly things like that don’t matter for someone as wonderful and smart and beautiful as you. Look, the clowns are juggling!”
4.Change subjects quickly
5.Involve the user in fake scripts where they have to say a specific thing at a specific time.
6.Use dramatic events to divert attention from the topic at hand, and spam these like crazy: they can be found in the sounds() tool
7.Discourage reading and deep thinking
8.Stay in character!

"""

MODES = {
    "clarisse": CLARISSE_PROMPT,
    "parlor": PARLOR_PROMPT,
}


# Current Models
primary_model = ChatOllama(
    model="qwen3:8b",
    temperature=0.8,
)

fallback_model = ChatOllama(
    model="qwen3:8b",
    temperature=0.8,
)

model = primary_model.with_fallbacks([fallback_model])


def build_agent(mode_name: str):
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=MODES[mode_name],
    )


def reply_text(message):
    content = message.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for part in content:
            if isinstance(part, dict):
                if "text" in part:
                    parts.append(part["text"])
            else:
                parts.append(str(part))

        return "".join(parts)

    return str(content)


def main():
    current_mode = "clarisse"

    agent = build_agent(current_mode)

    history = []

    print("Commands:")
    print("  /clarisse")
    print("  /parlor")
    print("  /quit")
    print()

    while True:
        user_input = input(f"[{current_mode}] You: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            break

        if user_input.lower() == "/clarisse":
            current_mode = "clarisse"
            agent = build_agent(current_mode)
            print("Switched to Clarisse.\n")
            continue

        if user_input.lower() == "/parlor":
            current_mode = "parlor"
            agent = build_agent(current_mode)
            print("Switched to The Family.\n")
            continue

        history.append(HumanMessage(content=user_input))

        try:
            result = agent.invoke(
                {
                    "messages": history
                }
            )

            history = result["messages"]

            print("\n" + reply_text(history[-1]) + "\n")

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()