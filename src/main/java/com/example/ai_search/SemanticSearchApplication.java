package com.example.ai_search;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;
import com.fasterxml.jackson.databind.json.JsonMapper;
import com.fasterxml.jackson.databind.util.StdDateFormat;
import com.github.victools.jsonschema.generator.*;
import com.github.victools.jsonschema.module.jackson.JacksonModule;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.converter.StructuredOutputConverter;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.ai.model.function.FunctionCallbackWrapper;
import org.springframework.ai.ollama.OllamaEmbeddingModel;
import org.springframework.ai.ollama.api.OllamaApi;
import org.springframework.ai.ollama.api.OllamaModel;
import org.springframework.ai.ollama.api.OllamaOptions;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Description;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.time.LocalDate;
import java.util.*;
import java.util.function.Function;

@Slf4j
@SpringBootApplication
public class SemanticSearchApplication {

    public static void main(String[] args) {
        SpringApplication.run(SemanticSearchApplication.class, args);
    }
}

// Configuration class for Ollama
@Configuration
class OllamaConfig {

    @Bean
    public OllamaApi ollamaApi() {
        return new OllamaApi();
    }

    @Bean
    public OllamaEmbeddingModel embeddingModel(OllamaApi ollamaApi) {
        // Set default Ollama options
        OllamaOptions defaultOptions = OllamaOptions.builder()
                .withModel(OllamaModel.MISTRAL.getName()) //  You need Models pre-trained for Tools support. For example mistral, firefunction-v2 or llama3.1:70b.
                .build();

        return new OllamaEmbeddingModel(ollamaApi, defaultOptions);
    }

    @Bean
    @Description("Get user by userId")
    public Function<Long, User> getUserByUserId(UserRepository userRepository) {
        return userRepository::getUserByUserId;
    }

    @Bean
    @Description("Get users by name")
    public Function<String, List<User>> getUsersByName(UserRepository userRepository) {
        return userRepository::getUsersByName;
    }

    @Bean
    @Description("Get project members by project name")
    public Function<String, List<User>> getProjectMembersByProjectName(ProjectRepository projectRepository) {
        return projectRepository::getProjectMembersByProjectName;
    }

    @Bean
    @Description("Get team members by team name")
    public Function<String, List<User>> getTeamMembersByTeamName(TeamRepository teamRepository) {
        return teamRepository::getTeamMembersByTeamName;
    }

    @Bean
    @Description("Get all MentionHistory that the given user mentions so far")
    public Function<Long, List<MentionHistory>> getMentionsByUser(MentionHistoryRepository mentionHistoryRepository) {
        return mentionHistoryRepository::getMentionsByUser;
    }
}

// Controller to handle the search API requests
@Slf4j
@RestController
@RequestMapping("/api/search")
class SearchController {

    @Autowired
    private SearchService searchService;

    @GetMapping
    public ResponseEntity<?> search(
            @RequestParam String query,
            @RequestParam(defaultValue = "5") int limit,
            @RequestParam(required = false) Long orgId,
            @RequestParam(required = false) Long teamId,
            @RequestParam(required = false) Long userId) {

        log.info("\n\n>> Received search request: query={}, limit={}, orgId={}, teamId={}, userId={}",
                query, limit, orgId, teamId, userId);

        try {
            List<User> results = searchService.search(query, limit, orgId, teamId, userId);
            return ResponseEntity.ok(results);
        } catch (Exception e) {
            log.error("Error occurred during search", e);
            return ResponseEntity.internalServerError().body("An error occurred during the search operation");
        }
    }
}

@Slf4j
@Service
class SearchService {

    private final UserRepository userRepository;
    private final ProjectRepository projectRepository;
    private final TeamRepository teamRepository;
    private final MentionHistoryRepository mentionHistoryRepository;
    private final ChatModel chatModel;

    @Autowired
    public SearchService(UserRepository userRepository,
                         ProjectRepository projectRepository,
                         TeamRepository teamRepository,
                         MentionHistoryRepository mentionHistoryRepository,
                         ChatModel chatModel) {
        this.userRepository = userRepository;
        this.projectRepository = projectRepository;
        this.teamRepository = teamRepository;
        this.mentionHistoryRepository = mentionHistoryRepository;
        this.chatModel = chatModel;
    }

    public List<User> search(String query, int limit, Long orgId, Long teamId, Long userId) {
        String contextualizedQuery = constructContextualizedQuery(query, orgId, teamId, userId);
        log.info("\n\n>> Contextualized query: {}", contextualizedQuery);

        UserMessage userMessage = new UserMessage(contextualizedQuery);
        var promptOptions = OllamaOptions.builder()
                .withModel(OllamaModel.LLAMA3_1)
                .withFunctionCallbacks(List.of(
                        FunctionCallbackWrapper.builder(this::getUserByUserIdWrapper)
                                .withName("getUserByUserId")
                                .withDescription("Get user by userId")
                                .withInputType(FunctionInputWrappers.UserIdWrapper.class)
                                .build(),
                        FunctionCallbackWrapper.builder(this::getUsersByNameWrapper)
                                .withName("getUsersByName")
                                .withDescription("Get users by name")
                                .withInputType(FunctionInputWrappers.NameWrapper.class)
                                .build(),
                        FunctionCallbackWrapper.builder(this::getProjectMembersByProjectNameWrapper)
                                .withName("getProjectMembersByProjectName")
                                .withDescription("Get project members by project name")
                                .withInputType(FunctionInputWrappers.ProjectNameWrapper.class)
                                .build(),
                        FunctionCallbackWrapper.builder(this::getTeamMembersByTeamNameWrapper)
                                .withName("getTeamMembersByTeamName")
                                .withDescription("Get team members by team name")
                                .withInputType(FunctionInputWrappers.TeamNameWrapper.class)
                                .build(),
                        FunctionCallbackWrapper.builder(this::getMentionsByUserWrapper)
                                .withName("getMentionsByUser")
                                .withDescription("Get all users that the given user mentions so far")
                                .withInputType(FunctionInputWrappers.UserIdWrapper.class)
                                .build()
                ))
                .build();

        ChatClient chatClient = ChatClient.builder(chatModel)
                .defaultSystem("""
                        You are a Search support agent called named "Eagle"."
                        Respond in a friendly, helpful, and joyful manner.
                        You are interacting with customers through an online chat system.
                        You are expected to provide information about users, projects, teams, and mentions.
                        When responding to a user query,
                        make sure you have the following information from the user: Name, UserId, OrgId, TeamId.
                        Check the message history for this information before asking the user.
                        Use the provided functions to fetch membership of user in teams, org, project and history of recent mentions if needed.
                        Use parallel function calling if required.
                        If you are unable to determine the information requested based on the provided parameters,don't suggest any user.
                        If your response does not contain any user information, return response as "No user found" and explain the reason.
                        If the query is very short, try to use getUsersByName function to find the user.
                        Today is {current_date}.
                        """)
                .defaultAdvisors(
//                        new PromptChatMemoryAdvisor(chatMemory), // Chat Memory
                        new LoggingAdvisor())
                .build();

        List<User> results = chatClient.prompt()
                .system(s -> s.param("current_date", LocalDate.now().toString()))
                .user(contextualizedQuery)
                .function("getUserByUserId", "Get user by userId", FunctionInputWrappers.UserIdWrapper.class, this::getUserByUserIdWrapper)
                .function("getUsersByName", "Get users by name", FunctionInputWrappers.NameWrapper.class, this::getUsersByNameWrapper)
                .function("getProjectMembersByProjectName", "Get project members by project name", FunctionInputWrappers.ProjectNameWrapper.class, this::getProjectMembersByProjectNameWrapper)
                .function("getTeamMembersByTeamName", "Get team members by team name", FunctionInputWrappers.TeamNameWrapper.class, this::getTeamMembersByTeamNameWrapper)
                .function("getMentionsByUser", "Get all users that the given user mentions so far", FunctionInputWrappers.UserIdWrapper.class, this::getMentionsByUserWrapper)
                .call()
                .entity(new GenericListOutputConverter<>(User.class));

        log.info("\n\n>> Structured response: {} \n\n", results);

        var response = chatModel.call(new Prompt(userMessage, promptOptions));
        log.info("\n\n>> normal response: {} \n\n", response.getResult().getOutput());

        List<User> filteredResults = filterResults(results, orgId, teamId, userId);

        return filteredResults.stream().limit(limit).toList();
    }

    private String constructContextualizedQuery(String query, Long orgId, Long teamId, Long userId) {
        StringBuilder contextualizedQuery = new StringBuilder(query);

        if (orgId != null) {
            contextualizedQuery.append(" within organization ID ").append(orgId);
        }
        if (teamId != null) {
            contextualizedQuery.append(" for team ID ").append(teamId);
        }
        if (userId != null) {
            contextualizedQuery.append(" relevant to user ID ").append(userId);
        }

        return contextualizedQuery.toString();
    }

    private List<User> processLLMResponse(ChatResponse response) {
        log.info("Raw LLM response: {}", response.getResult().getOutput());
        // TODO: Implement logic to extract user information from the LLM response
        // This might involve parsing the response content and using the appropriate repository methods
        log.warn("processLLMResponse method is not fully implemented");
        log.warn(response.toString());
        return new ArrayList<>(); // Placeholder
    }

    private List<User> filterResults(List<User> users, long orgId, long teamId, long userId) {
        ObjectMapper objectMapper = new ObjectMapper();
        List<User> userList = objectMapper.convertValue(users, new TypeReference<>() {
        });

        List<User> result = userList.stream()
                .filter(user -> orgId == 0 || user.orgId() == orgId)
                .toList();

        if (userList.size() != result.size()) {
            log.warn("Found some users that do not belong to the specified orgId");
        }

        return result;
    }

    // Function callback wrappers
    private User getUserByUserIdWrapper(FunctionInputWrappers.UserIdWrapper wrapper) {
        log.debug("Calling getUserByUserId with userId: {}", wrapper.userId);
        return userRepository.getUserByUserId(wrapper.userId);
    }

    private List<User> getUsersByNameWrapper(FunctionInputWrappers.NameWrapper wrapper) {
        log.debug("Calling getUsersByName with name: {}", wrapper.name);
        return userRepository.getUsersByName(wrapper.name);
    }

    private List<User> getProjectMembersByProjectNameWrapper(FunctionInputWrappers.ProjectNameWrapper wrapper) {
        log.debug("Calling getProjectMembersByProjectName with projectName: {}", wrapper.projectName);
        return projectRepository.getProjectMembersByProjectName(wrapper.projectName);
    }

    private List<User> getTeamMembersByTeamNameWrapper(FunctionInputWrappers.TeamNameWrapper wrapper) {
        log.debug("Calling getTeamMembersByTeamName with teamName: {}", wrapper.teamName);
        return teamRepository.getTeamMembersByTeamName(wrapper.teamName);
    }

    private List<MentionHistory> getMentionsByUserWrapper(FunctionInputWrappers.UserIdWrapper wrapper) {
        log.debug("Calling getMentionsByUser with userId: {}", wrapper.userId);
        return mentionHistoryRepository.getMentionsByUser(wrapper.userId);
    }
}

class FunctionInputWrappers {
    public static class UserIdWrapper {
        public Long userId;
    }

    public static class NameWrapper {
        public String name;
    }

    public static class ProjectNameWrapper {
        public String projectName;
    }

    public static class TeamNameWrapper {
        public String teamName;
    }
}

// Service for generating embeddings using Ollama
@Service
class OllamaEmbeddingService {

    private final OllamaEmbeddingModel embeddingModel;

    @Autowired
    public OllamaEmbeddingService(OllamaEmbeddingModel embeddingModel) {
        this.embeddingModel = embeddingModel;
    }

    public float[] getEmbedding(String text) {
        EmbeddingRequest request = new EmbeddingRequest(
                List.of(text),
                OllamaOptions.builder()
                        .withModel("chroma/all-minilm-l6-v2-f32")
                        .build()
        );

        // Make the API call to get the embedding
        EmbeddingResponse response = embeddingModel.call(request);

        // Check if embeddings were returned and extract them
        if (response.getResults() != null && !response.getResults().isEmpty()) {
            return response.getResults().get(0).getOutput();
        } else {
            return new float[0];  // Return empty array if no embeddings found
        }
    }
}

// A repository simulating a database of users
@Service
class UserRepository {
    private final List<User> users = List.of(
            new User(1, "John Doe", "john@techhub.com", 1, 10),
            new User(2, "Jane Smith", "jane.s@techhub.com", 1, 10),
            new User(3, "Robert Brown", "robert.brown@techhub.com", 1, 10),
            new User(4, "Emily Davis", "emily.davis@techhub.com", 1, 10),
            new User(5, "Michael Wilson", "michael.wilson@techhub.com", 2, 10),
            new User(6, "Sarah Johnson", "s.johnson@techhub.com", 2, 10),

            new User(7, "David Lee", "david.lee@globalcorp.com", 3, 20),
            new User(8, "Lisa Chen", "lisa.chen@globalcorp.com", 3, 20),
            new User(9, "James Taylor", "j.taylor@globalcorp.com", 3, 20),
            new User(10, "Emma White", "emma.white@globalcorp.com", 3, 20),
            new User(11, "Daniel Kim", "daniel.kim@globalcorp.com", 4, 20),
            new User(12, "Olivia Brown", "olivia.br@globalcorp.com", 4, 20),
            new User(13, "William Garcia", "william.garcia@globalcorp.com", 4, 20),

            new User(14, "Sophia Martinez", "sophia.martinez@innovatesoft.net", 5, 30),
            new User(15, "Alexander Rodriguez", "alexander.rodriguez@innovatesoft.net", 5, 30),
            new User(16, "Isabella Lopez", "isabella.lopez@innovatesoft.net", 5, 30),
            new User(17, "Ethan Hernandez", "ethan.hernandez@innovatesoft.net", 5, 30),
            new User(18, "Mia Gonzalez", "mia.gonzalez@innovatesoft.net", 5, 30),
            new User(19, "Benjamin Perez", "benjamin.perez@innovatesoft.net", 5, 30),
            new User(20, "Ava Sanchez", "ava.sanchez@innovatesoft.org", 5, 30),

            new User(21, "Christopher Torres", "christopher.torres@datawave.org", 6, 40),
            new User(22, "Amelia Flores", "amelia.flores@datawave.org", 6, 40),
            new User(23, "Matthew Ramirez", "matthew@datawave.org", 6, 40),
            new User(24, "Evelyn Rivera", "evelyn@datawave.org", 6, 40),
            new User(25, "Andrew Morales", "andrew.morales@datawave.org", 6, 40),

            new User(26, "Charlotte Ortiz", "charlotte.ortiz@nexustech.io", 7, 50),
            new User(27, "Joseph Cruz", "joseph.cruz@nexustech.io", 7, 50),
            new User(28, "Abigail Reyes", "abigail.rey@nexustech.io", 7, 50),
            new User(29, "Ryan Phillips", "ryan@nexustech.io", 7, 50),
            new User(30, "Elizabeth Campbell", "elizabeth.cam@nexustech.io", 7, 50)
    );

    public List<User> getUsersByName(String name) {
        return users.stream()
                .filter(user -> user.name().toLowerCase().contains(name.toLowerCase()) ||
                        user.email().toLowerCase().contains(name.toLowerCase()))
                .toList();
    }

    public User getUserByUserId(Long userId) {
        return users.stream()
                .filter(user -> user.userId() == userId)
                .findFirst()
                .orElse(null);
    }

    public List<User> getUsersByTeamId(Long teamId) {
        return users.stream()
                .filter(user -> user.teamId() == teamId)
                .toList();
    }
}

// A repository simulating a database of projects
@Service
class ProjectRepository {

    private static final Map<String, List<Long>> projects = Map.of(
            "Mars", Arrays.asList(3L, 1L, 2L),
            "Eagle Eye", Arrays.asList(4L, 5L));

    private final UserRepository userRepository;

    public ProjectRepository(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    public List<Long> findUsersByProjectName(String projectName) {
        return projects.getOrDefault(projectName, Collections.emptyList());
    }

    public List<User> getProjectMembersByProjectName(String prjName) {
        return projects.entrySet().stream()
                .filter(entry -> entry.getKey().toLowerCase().contains(prjName.toLowerCase()))
                .flatMap(entry -> entry.getValue().stream())

                .map(userRepository::getUserByUserId)
                .toList();
    }
}

// A repository simulating a database of Teams
@Service
class TeamRepository {

    private static final Map<Long, String> Teams = Map.of(
            1L,"Alpha",
            2L,"Beta");
    private final UserRepository userRepository;

    TeamRepository(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    public String getTeamNameById(Long teamId) {
        return Teams.get(teamId);
    }

    public List<User> getTeamMembersByTeamName(String teamName) {
        return Teams.entrySet().stream()
                .filter(entry -> entry.getValue().toLowerCase().contains(teamName.toLowerCase()))
                .map(Map.Entry::getKey)
                .map(userRepository::getUsersByTeamId)
                .flatMap(List::stream)
                .toList();
    }
}

// A repository simulating a database of Mentions
@Service
class MentionHistoryRepository {
    private static final List<MentionHistory> mentionHistory = new ArrayList<>(Arrays.asList(
            new MentionHistory(1, 2, LocalDate.now().minusDays(1)),
            new MentionHistory(1, 3, LocalDate.now().minusDays(2)),
            new MentionHistory(1, 5, LocalDate.now().minusDays(2)),
            new MentionHistory(2, 5, LocalDate.now().minusDays(1)),
            new MentionHistory(2, 6, LocalDate.now().minusWeeks(2)),
            new MentionHistory(26, 26, LocalDate.now().minusDays(2)),
            new MentionHistory(26, 27, LocalDate.now().minusDays(2)),
            new MentionHistory(26, 28, LocalDate.now().minusDays(2))
    ));

    private final UserRepository userRepository;

    MentionHistoryRepository(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    public List<MentionHistory> getMentionsByUser(long userId) {
        return mentionHistory.stream()
                .filter(mention -> mention.userId() == userId)
                .toList();
    }
}

class UserListOutputConverter implements StructuredOutputConverter<List<User>> {

    private final ObjectMapper objectMapper = new ObjectMapper();
    String listJsonSchema = """
            [
                {
                    "userId": "long",
                    "name": "string",
                    "email": "string",
                    "teamId": "long",
                    "orgId": "long"
                }
            ]
            """;

    @Override
    public List<User> convert(String json) {
        try {
            // Convert the JSON array into a List of User records
            return objectMapper.readValue(json, new TypeReference<List<User>>() {
            });
        } catch (IOException e) {
            throw new RuntimeException("Failed to convert JSON to List<User>. Json= " + json, e);
        }
    }

    public String getFormat() {
        String template = "Your response should be in JSON format.\n"
                + "Do not include any explanations, only provide a RFC8259 compliant JSON response following this format without deviation.\n"
                + "If your response does not contain any user information, return response \"No user found\"  and explain the reason in json format.\n" // multi response
                + "Do not include markdown code blocks in your response.\n"
                + "Do not include function names.\n"
                + "Do not include list name. only list of elements within the [] .\n"
                + "Remove the ```json markdown from the output.\nHere is the JSON Schema instance your output must adhere to:\n```%s```\n";
        return String.format(template, listJsonSchema);
    }
}

class GenericListOutputConverter<V> implements StructuredOutputConverter<List<V>> {
    private final ObjectMapper objectMapper;
    private final String jsonSchema;
    private final TypeReference<List<V>> typeRef;
    private final Class<V> valueType;

    public GenericListOutputConverter(Class<V> valueType) {
        this.objectMapper = createObjectMapper();
        this.typeRef = new TypeReference<>() {
        };
        this.jsonSchema = generateJsonSchemaForValueType(valueType);
        this.valueType = valueType;
    }

    @Override
    public List<V> convert(@NonNull String text) {
        try {
            String trimmedText = trimMarkdown(text);
            JsonNode jsonNode = objectMapper.readTree(trimmedText);

            // Check if the JSON is an object with a single array field
            if (jsonNode.isObject() && jsonNode.size() == 1) {
                JsonNode arrayNode = jsonNode.elements().next();
                if (arrayNode.isArray()) {
                    return objectMapper.convertValue(arrayNode, typeRef);
                }
            }

            // If it's not the special case, try to parse it as a direct array
            return objectMapper.readValue(trimmedText, typeRef);
        } catch (JsonProcessingException e) {
            throw new RuntimeException("Failed to convert JSON to List<V>. Json= " + text, e);
        }
    }

    @Override
    public String getFormat() {
        return String.format("""
                Your response should be in JSON format.
                The data structure for the JSON should be an object with a single field containing an array of %s objects.
                For example: {"items": [%s, %s, ...]}
                The array elements should adhere to this JSON Schema:
                ```
                %s
                ```
                Do not include any explanations, only provide a RFC8259 compliant JSON response following this format without deviation.
                """, valueType.getSimpleName(), getExampleJson(), getExampleJson(), this.jsonSchema);
    }

    private ObjectMapper createObjectMapper() {
        return JsonMapper.builder()
                .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
                .defaultDateFormat(new StdDateFormat().withColonInTimeZone(true))
                .build();
    }

    private String trimMarkdown(String text) {
        return text.replaceAll("^```json\\s*|\\s*```$", "").trim();
    }

    private String generateJsonSchemaForValueType(Class<V> valueType) {
        try {
            SchemaGeneratorConfig config = new SchemaGeneratorConfigBuilder(SchemaVersion.DRAFT_2020_12, OptionPreset.PLAIN_JSON)
                    .with(new JacksonModule())
                    .build();
            SchemaGenerator generator = new SchemaGenerator(config);
            JsonNode jsonNode = generator.generateSchema(valueType);

            ObjectWriter objectWriter = new ObjectMapper().writerWithDefaultPrettyPrinter();
            return objectWriter.writeValueAsString(jsonNode);
        } catch (JsonProcessingException e) {
            throw new RuntimeException("Could not generate JSON schema for value type: " + valueType.getName(), e);
        }
    }

    private String getExampleJson() {
        try {
            return objectMapper.writeValueAsString(objectMapper.createObjectNode());
        } catch (JsonProcessingException e) {
            return "{}";
        }
    }
}

//Model classes
record MentionHistory(long userId, long mentionedUserId, LocalDate timePeriod) {
}


