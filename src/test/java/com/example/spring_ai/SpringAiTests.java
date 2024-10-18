package com.example.spring_ai;

import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.PromptChatMemoryAdvisor;
import org.springframework.ai.chat.client.advisor.QuestionAnswerAdvisor;
import org.springframework.ai.chat.memory.InMemoryChatMemory;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.evaluation.EvaluationRequest;
import org.springframework.ai.evaluation.EvaluationResponse;
import org.springframework.ai.evaluation.RelevancyEvaluator;
import org.springframework.ai.model.Content;
import org.springframework.ai.model.Media;
import org.springframework.ai.ollama.api.OllamaModel;
import org.springframework.ai.ollama.api.OllamaOptions;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.core.io.ClassPathResource;
import org.springframework.util.MimeTypeUtils;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

@SpringBootTest
public class SpringAiTests {

	private static final Logger logger = LoggerFactory.getLogger(SpringAiTests.class);

	@Autowired
	private ChatModel chatModel;

	@Autowired
	private DataLoadingService dataLoadingService;

	@Autowired
	private VectorStore vectorStore;

	@Test
	void tellMeJoke() {

		ChatClient chatClient = ChatClient.create(chatModel);

		var response = chatClient.prompt()
				.user("Tell me a joke")
				.call()
				.content();

		logger.info("\n\n>> Response: {} \n\n", response);
	}

	@Test
	void tellMeJokeSystemPrompt() {

		ChatClient chatClient = ChatClient.builder(chatModel)
				.defaultSystem("You are a friendly assistant that answers question in the voice of {voice}.")
				.build();

		for (String voice : List.of("Pablo Picasso", "Elon Musk", "Shakespeare")) {

			var response = chatClient.prompt()
					.user("Tell me a joke")
					.system(sp -> sp.param("voice", voice))
					.call()
					.content();

			logger.info("\n\n>>  {} joke: {} \n\n", voice, response);
		}
	}

	/////////////////////
	// Structured output
	/////////////////////
	record ActorFilms(
			String actor,
			List<String> films) {
	}

	@Test
	void structuredOutput() {

		ChatClient chatClient = ChatClient.create(chatModel);

		ActorFilms response = chatClient.prompt()
				.user("Get the filmography for Tom Hanks")
				.call()
				.entity(ActorFilms.class);

		logger.info("\n\n>>  Response: {} \n\n", response);
	}

	@Test
	void structuredOutputList() {

		ChatClient chatClient = ChatClient.create(chatModel);

		List<ActorFilms> response = chatClient.prompt()
				.user("Generate the filmography for Tom Hanks and Robert DeNiro. Maximum 10 films")
				.call()
				.entity(new ParameterizedTypeReference<List<ActorFilms>>() {
				});

		logger.info("\n\n>> Response: {}", response);
	}

	/////////////////////
	// Prompt stuffing
	/////////////////////
	@Test
	void noContext() {
		var response = ChatClient.builder(chatModel).build()
				.prompt()
				.user("What is Carina?")
				.call()
				.content();

		logger.info("\n\n>> Response: {} \n\n", response);
	}

	@Test
	void stuffPrompt() {
		String context = """
				What is Carina?
				Founded in 2016, Carina is a technology nonprofit that provides a safe, easy-to-use, online
				location-based care matching service. We serve individuals and families searching for home
				care or child care and care professionals who are looking for good jobs. Carina is committed to
				building community and prioritizing people over profit. Through our partnerships with unions
				and social service agencies, we build online tools to bring good jobs to care workers, so they
				can focus on their passion — caring for others.
				Our vision is a care economy that strengthens our communities by respecting and supporting
				workers, individuals and families. We offer a care matching platform where verified care
				providers can connect with individuals and families who need care.
				""";

		var response = ChatClient.builder(chatModel).build()
				.prompt()
				.user("What is Carina? Please consider the following context when answering the question: "
						+ context)
				.call()
				.content();

		logger.info("\n\n>> Response: {} \n\n", response);
	}

	/////////////////////
	// RAG
	/////////////////////
	@Test
	void preLoadData() {
		dataLoadingService.load();
	}

	@Test
	void purposeQuestionWIthPreLoadedData() {

		String userText = "How does Carina work?";

		var response = ChatClient.builder(chatModel)
				.build().prompt()
				.advisors(new QuestionAnswerAdvisor(vectorStore, SearchRequest.defaults()))
				.user(userText)
				.call()
				.chatResponse();

		logger.info("\n\n>> Response: {} \n\n", response.getResult().getOutput().getContent());

		 evaluate(userText, response);
	}

	private void evaluate(String userText, ChatResponse response) {
		var relevancyEvaluator = new RelevancyEvaluator(ChatClient.builder(chatModel));

		EvaluationRequest evaluationRequest = new EvaluationRequest(userText,
				(List<Content>) response.getMetadata().get(QuestionAnswerAdvisor.RETRIEVED_DOCUMENTS),
				response.toString());

		EvaluationResponse evaluationResponse = relevancyEvaluator.evaluate(evaluationRequest);

		assertTrue(evaluationResponse.isPass(), "Response is not relevant to the question");
		System.out.println("Evaluation Test: " + evaluationResponse.isPass());
	}

	/////////////////////
	// Chat Memory
	/////////////////////
	@Test
	void conversationMemory() {

		var chatClient = ChatClient.builder(chatModel)
				.defaultAdvisors(new PromptChatMemoryAdvisor(new InMemoryChatMemory()))
//				 .defaultAdvisors(new MessageChatMemoryAdvisor(new InMemoryChatMemory()))
//				 .defaultAdvisors(new VectorStoreChatMemoryAdvisor(vectorStore))
				.build();

		var response = chatClient.prompt()
				.user("My name is John")
				.call()
				.content();

		logger.info("\n\n>> Response 1: {} \n\n", response);

		response = chatClient.prompt()
				.user("Please tell me, what is my name?")
				.call()
				.content();

		logger.info("\n\n>> Response 2: {} \n\n", response);
	}

	/////////////////////
	// Multi-modality
	/////////////////////
	@Test
	void multiModality() {
		var userMessage = new UserMessage("Explain what do you see on this picture?",
				new Media(MimeTypeUtils.IMAGE_PNG, new ClassPathResource("/data/test.png")));

		ChatResponse response = chatModel.call(new Prompt(userMessage,	OllamaOptions.builder().withModel(OllamaModel.LLAVA)));
        logger.info("\n\n>> Response2: {} \n\n", response.getResult().getOutput().getContent());

    }
}
