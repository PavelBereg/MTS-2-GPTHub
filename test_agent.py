import urllib.request
import json
import ssl

def test_chat():
    url = "http://localhost:8000/v1/chat/completions"
    
    payload = {
        "model": "mts-ai-assistant",
        "messages": [
            {
                "role": "user", 
                "content": "Привет! Расскажи коротко, что ты умеешь?"
            }
        ],
        "user": "hackaton_user_777"
    }
    
    data = json.dumps(payload).encode("utf-8")
    
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    
    print("Отправка запроса к ассистенту (через urllib)...")
    
    try:
        # Игнорируем проверку SSL для локальных тестов, если применимо
        context = ssl._create_unverified_context()
        
        with urllib.request.urlopen(req, context=context, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
            content = result['choices'][0]['message']['content']
            
            print("\n" + "="*50)
            print("ОТВЕТ АССИСТЕНТА:")
            print(content)
            print("="*50 + "\n")
            
            if "![" in content:
                print("Инструмент генерации изображений был упомянут/использован!")
            
    except Exception as e:
        print(f" Ошибка при тесте: {e}")
        print("\nПодсказка:")
        print("1. Проверьте, запущен ли контейнер: docker ps")
        print("2. Если не запущен, выполните: docker-compose up -d")

if __name__ == "__main__":
    test_chat()
