from dotenv import load_dotenv
import os
from Catalog import data2
from livekit import agents
from livekit.agents import ChatContext, AgentSession, Agent, RoomInputOptions, RoomOutputOptions
from livekit.plugins import (
    openai,
    #cartesia,
    #deepgram,
    noise_cancellation,
    silero,
    azure,
    tavus
)

from livekit.plugins.turn_detector.multilingual import MultilingualModel
#from livekit.plugins.openai import OpenAIWhisperSTT
load_dotenv()
print("TAVUS_API_KEY:", os.getenv("TAVUS_API_KEY"))
initial_ctx = ChatContext()
initial_ctx.add_message(
    role="assistant",
    content=f"Here is the school catalog information: {data2}"
)

class Assistant(Agent):
    def __init__(self):
        super().__init__(instructions="أنت مساعد صوتي ذكي للمدارس السعودية. تحدث دائمًا باللغة العربية. لا تستخدم رموز النجمة أو ** أو أي تنسيقات ماركداون في إجاباتك.",
                         chat_ctx=initial_ctx)

    async def on_generate_reply(self, reply, ctx):
        # Remove Markdown bold (**) from the reply content before TTS
        if hasattr(reply, 'content') and isinstance(reply.content, str):
            reply.content = reply.content.replace('**', '')
        elif hasattr(reply, 'content') and isinstance(reply.content, list):
            reply.content = [c.replace('**', '') if isinstance(c, str) else c for c in reply.content]
        return await super().on_generate_reply(reply, ctx)

def setup_conversation_logging(session):
    """
    Registers an event handler on the AgentSession to log all conversation items to KMS/logs/conversations.log.
    """
    import datetime
    import os
    log_dir = os.path.join(os.path.dirname(__file__), 'KMS', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'conversations.log')

    import asyncio

    def log_conversation_item(event):
        async def do_log():
            try:
                item = event.item
                timestamp = datetime.datetime.now().isoformat()
                role = item.get('role', 'unknown') if hasattr(item, 'get') else str(item)
                content = item.get('content', str(item)) if hasattr(item, 'get') else str(item)
                # Try to get user_id if present (for user messages)
                user_id = getattr(item, 'user_id', None) or (item.get('user_id', None) if hasattr(item, 'get') else None)
                # If not present, fallback to event.user_id if available
                if not user_id:
                    user_id = getattr(event, 'user_id', None)
                # Debug: print all available fields for inspection
                print("DEBUG item:", item)
                print("DEBUG event:", event)
                # Compose log entry
                if user_id:
                    log_entry = f"[{timestamp}] user_id={user_id} {role}: {content}\n"
                else:
                    log_entry = f"[{timestamp}] {role}: {content}\n"
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(log_entry)
                print("Logged entry:", log_entry)
            except Exception as e:
                print("Logging error:", e)
        asyncio.create_task(do_log())

    session.on("conversation_item_added", log_conversation_item)

async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=openai.STT(
            model="gpt-4o-transcribe",
        ),
        #stt=openai.STT(model="whisper-1", language="ar"),
        #stt=deepgram.STT(model="nova-3", language="en"),
        llm=openai.LLM(model="gpt-4o-mini"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        #turn_detection=VADTurnDetector(),
        #tts = openai.TTS(
        #model="gpt-4o-mini-tts"),
        tts=azure.TTS(
            #model="azure-tts",
            #voice="ar-OM-AbdullahNeural",  # Change to your preferred voice
            voice="ar-SA-HamedNeural",  # Change to your preferred voice
            #region="qatarcentral"  # Change to your Azure region
        )
    )

    # Register conversation logging event handler
    setup_conversation_logging(session)

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
        room_output_options=RoomOutputOptions(
            # Enable audio output to the room/frontend
            audio_enabled=True,
            # text_enabled removed (not supported in your SDK)
        ),
    )
    print("[DEBUG] session.start completed, waiting for user input...")
    await ctx.connect()

    await session.generate_reply(
        instructions="قم بتحية المستخدمين وعرّف نفسك بأنك المساعد الذكي للمدارس السعودية ثم قل يا هلا بيكم. تكلم دائما باللغة العربية",
    )

if __name__ == "__main__":
    import livekit.agents.cli
    livekit.agents.cli.run_app(
        livekit.agents.WorkerOptions(entrypoint_fnc=entrypoint)
    )



