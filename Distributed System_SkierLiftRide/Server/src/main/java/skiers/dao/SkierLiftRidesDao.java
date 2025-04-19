package skiers.dao;

import jakarta.annotation.PreDestroy;
import org.springframework.stereotype.Component;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.dynamodb.DynamoDbClient;
import software.amazon.awssdk.services.dynamodb.model.*;

import java.util.List;
import java.util.Map;

@Component
public class SkierLiftRidesDao {
  private final DynamoDbClient dynamoDbClient;
  private static final String TABLE_NAME = "SkierLiftRides";

  public SkierLiftRidesDao() {
    this.dynamoDbClient = DynamoDbClient.builder()
        .region(Region.US_WEST_2)
        .build();
  }

  @PreDestroy
  public void shutdown() {
    dynamoDbClient.close();
  }


  public List<Map<String, AttributeValue>> queryBySkierId(int skierId) {
    QueryRequest queryRequest = QueryRequest.builder()
        .tableName(TABLE_NAME)
        .keyConditionExpression("SkierID = :id")
        .expressionAttributeValues(Map.of(
            ":id", AttributeValue.builder().n(String.valueOf(skierId)).build()
        ))
        .build();

    try {
      QueryResponse response = dynamoDbClient.query(queryRequest);
      return response.items();
    } catch (DynamoDbException e) {
      System.err.println("Query failed: " + e.getMessage());
      return List.of();
    }
  }

}
