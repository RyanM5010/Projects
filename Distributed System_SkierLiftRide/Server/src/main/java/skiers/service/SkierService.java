package skiers.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import commons.shared.model.LiftRide;
import software.amazon.awssdk.services.dynamodb.model.AttributeValue;
import skiers.dao.*;
import java.util.List;
import java.util.Map;

@Service
public class SkierService {

  private final SkierLiftRidesDao skierLiftRidesDao;
  private final LiftRideCountsDao countsDao;

  @Autowired
  public SkierService(SkierLiftRidesDao skierLiftRidesDao, LiftRideCountsDao countsDao) {
    this.skierLiftRidesDao = skierLiftRidesDao;
    this.countsDao = countsDao;
  }

  public boolean isUrlValid(int resortId, int seasonId, int dayId, int skierId) {
    return resortId > 0 &&
        seasonId > 0 &&
        dayId >= 1 &&
        dayId <= 366 &&
        skierId > 0;
  }

  public boolean isLiftRideValid(LiftRide liftRide) {
    if (liftRide == null) {
      return false;
    }

    Integer time = liftRide.getTime();
    Integer liftID = liftRide.getLiftID();

    if (time == null || liftID == null) {
      return false;
    }

    return isInteger(time) && isInteger(liftID) &&
        time >= 1 && time <= 360 &&
        liftID >= 1 && liftID <= 40;
  }

  public int getTotalLiftRideCount(int resortId, int seasonId, int dayId) {
    String key = resortId + "#" + seasonId + "#" + dayId;
    return countsDao.getLiftRideCounts(key);
  }

  public int getTotalVertical(int skierId) {
    List<Map<String, AttributeValue>> items =
        skierLiftRidesDao.queryBySkierId(skierId);
    int total = 0;
    for (Map<String, AttributeValue> item : items) {
      int liftId = Integer.parseInt(item.get("LiftID").n());
      total += liftId * 10;
    }
    return total;
  }

  private boolean isInteger(Number value) {
    return value instanceof Integer;
  }
}