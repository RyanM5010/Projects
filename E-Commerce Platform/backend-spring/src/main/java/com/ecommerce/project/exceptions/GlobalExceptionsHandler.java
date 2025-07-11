package com.ecommerce.project.exceptions;
import com.ecommerce.project.payload.APIResponse;
import java.util.HashMap;
import java.util.Map;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice
public class GlobalExceptionsHandler {

  @ExceptionHandler(MethodArgumentNotValidException.class)
  public ResponseEntity<Map<String, String>> myMethodArgumentNotValidException(MethodArgumentNotValidException e){
    Map<String, String> response = new HashMap<>();
    e.getBindingResult().getAllErrors().forEach((error) -> {
      String fieldName = ((FieldError) error).getField();
      String errorMessage = error.getDefaultMessage();
      response.put(fieldName, errorMessage);
    });
    return new ResponseEntity<Map<String, String>>(response,
        HttpStatus.BAD_REQUEST);
  }
  @ExceptionHandler(ResourceNotFoundException.class)
  public ResponseEntity<APIResponse> myResourceNotFoundException(ResourceNotFoundException e) {
    String message = e.getMessage();
    APIResponse response = new APIResponse(message, false);
    return new ResponseEntity<>(response, HttpStatus.NOT_FOUND);
  }

  @ExceptionHandler(APIException.class)
  public ResponseEntity<APIResponse> myAPIException(APIException e) {
    String Message = e.getMessage();
    APIResponse response = new APIResponse(Message, false);
    return new ResponseEntity<>(response, HttpStatus.BAD_REQUEST);
  }
}
