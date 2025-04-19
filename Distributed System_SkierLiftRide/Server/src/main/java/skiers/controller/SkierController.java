package skiers.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import commons.shared.model.LiftRide;
import skiers.model.ResponseMsg;
import skiers.service.SkierService;
import skiers.service.rabbitmq.RabbitMQService;



@RestController
@RequestMapping("/skiers")
public class SkierController {

  private final SkierService skierService;
  private final RabbitMQService rabbitMQService;

  public SkierController(SkierService skierService, RabbitMQService rabbitMQService) {
    this.skierService = skierService;
    this.rabbitMQService = rabbitMQService;
  }


  // 2. GET total vertical for a skier
  @GetMapping("/{skierId}/vertical")
  public ResponseEntity<ResponseMsg> getTotalVertical(@PathVariable int skierId) {
    int totalVertical = skierService.getTotalVertical(skierId);
    return ResponseEntity.ok(new ResponseMsg("Total vertical: " + totalVertical));
  }

  @PostMapping("/{resortId}/seasons/{seasonId}/days/{dayId}/skiers/{skierId}")
  public ResponseEntity<ResponseMsg> postLiftRide(
      @PathVariable int resortId,
      @PathVariable int seasonId,
      @PathVariable int dayId,
      @PathVariable int skierId,
      @RequestBody LiftRide liftRide) {

    // URL validation
    if (!skierService.isUrlValid(resortId, seasonId, dayId, skierId)) {
        return ResponseEntity.badRequest()
            .body(new ResponseMsg("Invalid URL parameters"));
    }

    try {
        // Payload validation
        if (liftRide == null) {
            return ResponseEntity.badRequest()
                .body(new ResponseMsg("Missing request body data"));
        }

        // Value range validation
        if (!skierService.isLiftRideValid(liftRide)) {
            return ResponseEntity.badRequest()
                .body(new ResponseMsg("Invalid lift ride data"));
        }

        // Send to RabbitMQ
        rabbitMQService.sendLiftRideMessage(resortId, seasonId, dayId, skierId, liftRide);

        return ResponseEntity.status(201)
            .body(new ResponseMsg("Write successful"));

    } catch (Exception e) {
        return ResponseEntity.badRequest()
            .body(new ResponseMsg("Invalid inputs: " + e.getMessage()));
    }
  }
}