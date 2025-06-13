import os
from typing import Dict, List, Optional, Union

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, Tool


class GeminiClient:
    """
    Uma classe para facilitar a integração com a API do Google Gemini.

    Permite a geração de texto e interações de chat, abstraindo a configuração
    e o uso direto do SDK do Gemini.

    Modelos Gemini recomendados hoje (consulte a documentação oficial para os mais recentes):
    - gemini-1.5-flash: Modelo multimodal rápido e versátil, otimizado para escalabilidade.
                       Bom para a maioria das tarefas gerais.
    - gemini-1.5-pro: Modelo multimodal de tamanho médio, otimizado para uma ampla gama
                     de tarefas de raciocínio. Mais capaz para tarefas complexas.
    - gemini-2.0-flash (Preview): Nova geração, com foco em velocidade e eficiência.
                                  Ideal para baixa latência e alto volume.
    - gemini-2.5-pro-preview-05-06 (Preview): A mais poderosa capacidade de raciocínio da nova geração. Melhor para codificação complexa e análise de grandes bases de dados.
    - Outras variaçõe: gemini-2.0-flash-preview-image-generation, gemini-2.0-flash-lite, gemini-2.5-pro-preview-05-06, gemini-2.5-flash-preview-04-17, gemini-2.5-flash-preview-05-20
    - Modelos Familia Gemma: gemma-3-27b-it, gemma-3-4b-it, gemma-3-12b-it
    """

    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gemini-1.5-flash",  # Modelo base recomendado hoje
        temperature: float = 0.9,
        max_output_tokens: int = 8192,  # Aumentado para o limite do 1.5 Flash/Pro
        top_p: float = 1.0,
        top_k: int = 32,
        system_instruction: Optional[str] = None,
        safety_settings: Optional[List[Dict]] = None,
    ):
        """
        Inicializa o cliente Gemini.

        Args:
            api_key (str, optional): A chave da API do Gemini. Se não for fornecida, a classe tentará carregá-la da variável de ambiente 'GOOGLE_API_KEY'.
            model_name (str, optional): O nome do modelo Gemini a ser usado. Padrão é "gemini-1.5-flash". Outras opções incluem "gemini-1.5-pro", "gemini-2.0-flash", etc. Consulte a documentação para os modelos mais recentes.
            temperature (float): Controla a aleatoriedade da saída. Valores mais altos (0.0-1.0) produzem resultados mais criativos, mas potencialmente menos coerentes. Padrão: 0.9 (um pouco mais criativo).
            max_output_tokens (int): O número máximo de tokens (palavras/partes) que o modelo pode gerar. Padrão: 8192 (limite para modelos 1.5 e 2.0 Flash/Pro).
            top_p (float): Amostragem com núcleo. O modelo considera tokens cuja probabilidade soma até este valor. Padrão: 1.0 (considera todos os tokens).
            top_k (int): Amostragem de top-k. O modelo considera os top-k tokens mais prováveis. Padrão: 32.
            system_instruction (str, optional): Uma instrução de sistema que guia o comportamento geral do modelo. Define um persona ou um conjunto de regras para todas as interações.
            safety_settings (List[Dict], optional): Configurações de segurança para ajustar os limites de conteúdo inseguro. Por padrão, utiliza as configurações padrão do Gemini.
        Raises:
            ValueError: Se a chave da API não for fornecida e não for encontrada nas variáveis de ambiente.
        """
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "A chave da API do Gemini não foi fornecida e não foi encontrada "
                    "na variável de ambiente 'GOOGLE_API_KEY'. "
                    "Por favor, forneça a chave ou defina a variável de ambiente."
                )

        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.default_generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            top_k=top_k,
        )
        self.system_instruction = system_instruction
        self.safety_settings = safety_settings if safety_settings is not None else []
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.default_generation_config,
            safety_settings=self.safety_settings,
            system_instruction=self.system_instruction,
        )
        self.chat_session = None

    def generate_response(
        self,
        prompt: Union[
            str, List[Union[str, bytes, Dict]]
        ],  # Adicionado Dict para compatibilidade com partes de conteúdo mais complexas
        generation_config: Optional[GenerationConfig] = None,
        tools: Optional[List[Tool]] = None,
        stream: bool = False,
    ) -> Union[str, bytes, Dict, List[Union[str, bytes, Dict]]]:
        """
        Gera uma resposta de texto padrão a partir de um prompt.

        Args:
            prompt (Union[str, List[Union[str, bytes, Dict]]]): O prompt de entrada para o modelo. Pode ser uma string (para texto) ou uma lista de strings/bytes/Dict (para multimodal, e.g., texto e imagem, ou objetos de tool_code).
            generation_config (GenerationConfig, optional): Configurações de geração específicas para esta chamada, substituindo as configurações padrão do cliente.
            tools (List[Tool], optional): Uma lista de ferramentas (funções) que o modelo pode usar.
            stream (bool): Se True, a resposta será retornada como um iterador que gera chunks;Se False, a resposta completa é retornada de uma vez.

        Returns:
            Union[str, bytes, Dict, List[Union[str, bytes, Dict]]]: O conteúdo gerado pelo modelo. Retorna uma string para texto simples, ou um objeto mais complexo para outros tipos de saída (e.g., JSON se configurado). Se `stream` for True, retorna um iterador de strings.
        """
        # A configuração específica da chamada sobrepõe a configuração padrão do modelo,
        # e o system_instruction já está definido na instância do modelo.
        # Se for fornecido um generation_config aqui, ele será usado.
        effective_config = generation_config if generation_config else self.default_generation_config

        try:
            response = self.model.generate_content(
                contents=prompt,
                generation_config=effective_config,
                tools=tools,
                stream=stream,
            )
            if stream:
                # Retorna um gerador de texto
                return (chunk.text for chunk in response)
            else:
                # Retorna o texto completo, tratando casos onde 'text' não está presente
                # mas há candidatos com partes de conteúdo (ex: tool_code)
                if hasattr(response, "text"):
                    return response.text
                elif response.candidates and response.candidates[0].content.parts:
                    # Tenta retornar a primeira parte do primeiro candidato se houver
                    first_part = response.candidates[0].content.parts[0]
                    return (
                        first_part.text if hasattr(first_part, "text") else first_part
                    )  # Retorna o objeto Part se não for texto
                else:
                    return ""  # Retorna string vazia se não houver texto nem partes
        except Exception as e:
            print(f"Erro ao gerar resposta: {e}")
            return f"Erro: {e}"

    def generate_response_instructed(
        self,
        prompt: Union[str, List[Union[str, bytes, Dict]]],
        instruction: str,
        generation_config: Optional[GenerationConfig] = None,
        tools: Optional[List[Tool]] = None,
        stream: bool = False,
    ) -> Union[str, bytes, Dict, List[Union[str, bytes, Dict]]]:
        """
        Gera uma resposta de texto com um comportamento guiado por uma instrução específica.
        Esta instrução temporária **substitui** o `system_instruction` global
        do cliente para esta chamada, criando uma nova instância de modelo temporária.

        Args:
            prompt (Union[str, List[Union[str, bytes, Dict]]]): O prompt de entrada para o modelo.
            instruction (str): A instrução específica para guiar o comportamento do modelo nesta interação (e.g., "Responda como um pirata.").
            generation_config (GenerationConfig, optional): Configurações de geração específicas para esta chamada. Se omitido, usa as configurações padrão do cliente.
            tools (List[Tool], optional): Uma lista de ferramentas (funções) que o modelo pode usar.
            stream (bool): Se True, a resposta será retornada como um iterador que gera chunks.

        Returns:
            Union[str, bytes, Dict, List[Union[str, bytes, Dict]]]: O conteúdo gerado pelo modelo.
        """
        # Para aplicar uma instrução temporária, precisamos criar uma nova instância de GenerativeModel
        # ou ajustar o system_instruction temporariamente, o que é mais limpo com uma nova instância.
        # Caso contrário, o system_instruction original do objeto self.model seria usado.

        # Clona a configuração padrão ou usa a fornecida para evitar modificações indesejadas
        effective_config = (
            generation_config if generation_config else GenerationConfig(**self.default_generation_config.to_dict())
        )

        # Cria uma nova instância de modelo com a instrução específica
        instructed_model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=effective_config,
            safety_settings=self.safety_settings,
            system_instruction=instruction,  # A instrução específica para esta chamada
        )

        try:
            response = instructed_model.generate_content(
                contents=prompt,
                tools=tools,
                stream=stream,
            )
            if stream:
                return (chunk.text for chunk in response)
            else:
                if hasattr(response, "text"):
                    return response.text
                elif response.candidates and response.candidates[0].content.parts:
                    first_part = response.candidates[0].content.parts[0]
                    return first_part.text if hasattr(first_part, "text") else first_part
                else:
                    return ""
        except Exception as e:
            print(f"Erro ao gerar resposta instruída: {e}")
            return f"Erro: {e}"

    def start_chat(self, history: Optional[List[Dict]] = None):
        """
        Inicia uma sessão de chat com o modelo.

        Args:
            history (List[Dict], optional): Histórico de mensagens para inicializar o chat.
            Formato: [
                {"role": "user", "parts": ["..."]},
                {"role": "model", "parts": ["..."]}
            ]
        """
        # O chat_session é iniciado a partir da instância do modelo existente
        self.chat_session = self.model.start_chat(history=history)
        print(f"Sessão de chat iniciada com o modelo: {self.model_name}")

    def send_chat_message(
        self,
        message: Union[str, List[Union[str, bytes, Dict]]],
        stream: bool = False,
    ) -> Union[str, List[Dict]]:
        """
        Envia uma mensagem para a sessão de chat e obtém uma resposta.

        Args:
            message (Union[str, List[Union[str, bytes, Dict]]]): A mensagem a ser enviada.
            stream (bool): Se True, a resposta será retornada como um iterador.

        Returns:
            Union[str, List[Dict]]: O conteúdo da resposta do modelo.
                                    Se `stream` for True, retorna um iterador.
        Raises:
            RuntimeError: Se uma sessão de chat não foi iniciada.
        """
        if not self.chat_session:
            raise RuntimeError("Nenhuma sessão de chat iniciada. Chame 'start_chat()' primeiro.")

        try:
            response = self.chat_session.send_message(message, stream=stream)
            if stream:
                return (chunk.text for chunk in response)
            else:
                if hasattr(response, "text"):
                    return response.text
                elif response.candidates and response.candidates[0].content.parts:
                    first_part = response.candidates[0].content.parts[0]
                    return first_part.text if hasattr(first_part, "text") else first_part
                else:
                    return ""
        except Exception as e:
            print(f"Erro ao enviar mensagem no chat: {e}")
            return f"Erro: {e}"


# Exemplo de uso:
if __name__ == "__main__":
    # Certifique-se de definir sua GOOGLE_API_KEY como uma variável de ambiente
    # Ex: export GOOGLE_API_KEY='SUA_CHAVE_AQUI'
    # Ou passe-a diretamente: client = GeminiClient(api_key='SUA_CHAVE_AQUI')

    try:
        # 1. Cliente com configurações padrão (modelo gemini-1.5-flash)
        print("--- Teste de Geração de Texto Padrão ---")
        gemini_client = GeminiClient()
        response_text = gemini_client.generate_response("Qual é a capital do Brasil?")
        print(f"Resposta padrão: {response_text}\n")

        # 2. Cliente com modelo e temperatura específicos
        print("--- Teste de Geração de Texto com Modelo e Temperatura ---")
        gemini_pro_client = GeminiClient(model_name="gemini-1.5-pro", temperature=0.5)
        response_pro = gemini_pro_client.generate_response("Escreva uma pequena história sobre um gato astronauta.")
        print(f"Resposta com gemini-1.5-pro (temp=0.5): {response_pro}\n")

        # 3. Geração de texto com instrução
        print("--- Teste de Geração de Texto com Instrução ---")
        response_instructed = gemini_client.generate_response_instructed(
            "Crie um slogan para um novo tipo de refrigerante de limão.",
            "Responda de forma divertida e espirituosa, com um tom de marketing.",
        )
        print(f"Resposta instruída: {response_instructed}\n")

        # 4. Geração de texto com streaming
        print("--- Teste de Geração de Texto com Streaming ---")
        print("Resposta em streaming:")
        stream_response_generator = gemini_client.generate_response(
            "Me explique sobre buracos negros em 3 parágrafos.", stream=True
        )
        for chunk in stream_response_generator:
            print(chunk, end="", flush=True)
        print("\n")

        # 5. Teste de sessão de chat
        print("--- Teste de Sessão de Chat ---")
        chat_client = GeminiClient(
            model_name="gemini-1.5-flash", system_instruction="Você é um assistente de vendas educado e prestativo."
        )
        chat_client.start_chat()
        chat_response1 = chat_client.send_chat_message("Olá, estou interessado em um novo smartphone.")
        print(f"Chat (Usuário): Olá, estou interessado em um novo smartphone.")
        print(f"Chat (Modelo): {chat_response1}")

        chat_response2 = chat_client.send_chat_message("Quais são as melhores opções para fotografia?")
        print(f"Chat (Usuário): Quais são as melhores opções para fotografia?")
        print(f"Chat (Modelo): {chat_response2}\n")

    except ValueError as e:
        print(f"Erro de configuração: {e}")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
