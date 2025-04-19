import io.swagger.client.model.LiftRide;
import java.util.concurrent.ThreadLocalRandom;

public class DataGenerator {
  private static final int MIN_SKIER_ID = 1;
  private static final int MAX_SKIER_ID = 100000;
//    private static final int MIN_RESORT_ID = 1;
//    private static final int MAX_RESORT_ID = 10;
  private static final int RESORT_ID = 1;
  private static final int MIN_LIFT_ID = 1;
  private static final int MAX_LIFT_ID = 40;
  private static final int MIN_TIME = 1;
  private static final int MAX_TIME = 360;
  private static final String SEASON_ID = "2025";

  public static LiftRideEvent generateLiftRide(String dayId) {
    int skierID = ThreadLocalRandom.current().nextInt(MIN_SKIER_ID, MAX_SKIER_ID + 1);
//    int resortID = ThreadLocalRandom.current().nextInt(MIN_RESORT_ID, MAX_RESORT_ID + 1);
    int liftID = ThreadLocalRandom.current().nextInt(MIN_LIFT_ID, MAX_LIFT_ID + 1);
    int time = ThreadLocalRandom.current().nextInt(MIN_TIME, MAX_TIME + 1);

    LiftRide liftRide = new LiftRide();
    liftRide.setLiftID(liftID);
    liftRide.setTime(time);

    return new LiftRideEvent(RESORT_ID, skierID, SEASON_ID, dayId, liftRide);
  }
}
