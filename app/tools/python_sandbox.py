import asyncio
import os
import uuid
from typing import Annotated
from langchain_core.tools import InjectedToolCallId, tool
from langchain_core.messages import ToolMessage
import logging
logger = logging.getLogger("mts.sandbox")

@tool
async def execute_python(
    code: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> ToolMessage:
    """Tool for executing Python code locally. Use this for data analysis, pandas operations, and plotting matplotlib charts. 
    CRITICAL RULES FOR DATA ANALYSIS (EDA) AND PLOTTING:
    1. STOP TALKING, START CODING: If the user asks for analysis, EDA, or a chart, you are STRICTLY FORBIDDEN from just describing the data in text. You MUST actively write and execute Python code.
    2. SHOW, DON'T TELL: Do not list percentages or statistics in plain text if you were asked for a chart. Write a script to calculate and plot them.
    3. HOW TO LOAD DATA: You do not have physical files. Hardcode the data from context as a raw string and read using `io.StringIO()`.
    4. MANDATORY PLOTTING: You MUST save the plot to disk using `plt.savefig('static/downloads/filename.png')`.
    5. MANDATORY OUTPUT: Your final response MUST contain the absolute Markdown link: `![График](http://localhost:8000/static/downloads/filename.png)`.

    Example of Expected Behavior:
    User: Сделай EDA и нарисуй график по этим данным: [данные]
    Assistant (Thinking): I must not just summarize this. I must write a Python script.
    Assistant (Action): Calls `execute_python` with script using `io.StringIO` and `plt.savefig`.
    """
    import os
    import sys
    import uuid
    from app.core.config import get_settings
    
    settings = get_settings()
    
    # 1. Удали маркдаун-обертки
    if code.startswith('```python'):
        code = code[9:]
    elif code.startswith('```'):
        code = code[3:]
    if code.endswith('```'):
        code = code[:-3]
    code = code.strip()

    # 2. Принудительно добавь заголовок
    header = f"""import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import uuid
os.makedirs('static/downloads', exist_ok=True)
"""
    full_code = header + code

    # 3. Сохрани итоговый код во временный файл
    import uuid
    file_name = f"temp_exec_{uuid.uuid4().hex}.py"
    
    try:
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(full_code)

        # 4. Запусти файл через asyncio.create_subprocess_exec
        import sys
        process = await asyncio.create_subprocess_exec(
            sys.executable, file_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # 5. Оберни запуск в asyncio.wait_for с таймаутом 15 секунд
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=15.0)
            out = stdout.decode('utf-8')
            err = stderr.decode('utf-8')
            
            if out:
                print(f"--- SANDBOX OUT ---\n{out}\n-------------------")
            if err:
                print(f"--- SANDBOX ERR ---\n{err}\n-------------------")
                
            result = out + err
            
            # Временный дебаг-лог файл
            with open("sandbox_debug.log", "a", encoding="utf-8") as debug_f:
                debug_f.write(f"\n--- {uuid.uuid4().hex} ---\nCODE:\n{code}\nOUT:\n{out}\nERR:\n{err}\n")
                
            return ToolMessage(content=result, tool_call_id=tool_call_id, name="execute_python")
        except asyncio.TimeoutError:
            try:
                process.kill()
            except Exception:
                pass
            return ToolMessage(content="Error: Execution timed out (15s limit)", tool_call_id=tool_call_id, name="execute_python")
    finally:
        # 6. Удали временный файл в блоке finally
        if os.path.exists(file_name):
            try:
                os.remove(file_name)
            except OSError:
                pass
