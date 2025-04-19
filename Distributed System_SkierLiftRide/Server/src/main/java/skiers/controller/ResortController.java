package skiers.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import commons.shared.model.LiftRide;
import skiers.model.ResponseMsg;
import skiers.service.SkierService;
import skiers.service.rabbitmq.RabbitMQService;

@RestController
@RequestMapping("/resorts")
public class ResortController {
  private final SkierService skierService;

  public ResortController(SkierService skierService) {
    this.skierService = skierService;
  }

  // 1. GET total lift rides for a day at a resort
  @GetMapping("/{resortId}/seasons/{seasonId}/day/{dayId}/skiers")
  public ResponseEntity<ResponseMsg> getLiftRideCount(
      @PathVariable int resortId,
      @PathVariable int seasonId,
      @PathVariable int dayId) {

    int count = skierService.getTotalLiftRideCount(resortId, seasonId, dayId);
    return ResponseEntity.ok(new ResponseMsg("Total lift rides: " + count));
  }

}
