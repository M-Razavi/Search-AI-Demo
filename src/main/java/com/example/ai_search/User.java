package com.example.ai_search;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Builder;

import java.time.LocalDateTime;

@Builder
public record User(
        @JsonProperty("userId") long userId,
        @JsonProperty("name") String name,
        @JsonProperty("email") String email,
        @JsonProperty("teamId") long teamId,
        @JsonProperty("orgId") long orgId
) {
    @JsonCreator
    public User {
    }
}