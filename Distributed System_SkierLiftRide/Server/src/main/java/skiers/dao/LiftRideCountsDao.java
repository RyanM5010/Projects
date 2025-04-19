package skiers.dao;

import jakarta.annotation.PreDestroy;
import org.springframework.stereotype.Component;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.dynamodb.DynamoDbClient;
import software.amazon.awssdk.services.dynamodb.model.*;

import java.util.Map;

@Component
public class LiftRideCountsDao {
  private final DynamoDbClient dynamoDbClient;
  private static final String TABLE_NAME = "LiftRideCounts";

  public LiftRideCountsDao() {
    this.dynamoDbClient = DynamoDbClient.builder()
        .region(Region.US_WEST_2)
        .build();
  }

  @PreDestroy
  public void shutdown() {
    dynamoDbClient.close();
  }

  public int getLiftRideCounts(String resortSeasonDayId) {
    GetItemRequest request = GetItemRequest.builder()
        .tableName(TABLE_NAME)
        .key(Map.of(
            "ResortSeasonDayID", AttributeValue.builder().s(resortSeasonDayId).build()
        ))
        .attributesToGet("count")
        .build();

    try {
      Map<String, AttributeValue> item = dynamoDbClient.getItem(request).item();
      if (item != null && item.containsKey("count")) {
        return Integer.parseInt(item.get("count").n());
      }
    } catch (DynamoDbException e) {
      System.err.println("Failed to get count: " + e.getMessage());
    }
    return 0;
  }
}
