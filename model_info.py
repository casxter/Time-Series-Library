from torchinfo import summary
import models.TimesNet,models.TimeMixer

summary(models.TimesNet.Model())
summary(models.TimeMixer.Model())